"""Deterministic synthetic UPI/IMPS transaction generator for RupeeTrace.

Run:  python -m data.generator            (writes data/out/*.csv)
      python -m data.generator --n-accounts 1000 --out /tmp/x   (small run)

Ground-truth columns (label, typology, scenario_id, root_txn_id_gt, role_gt) are
evaluation-only. Nothing in this module reads them back as model input.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from data import schemas as S

T0 = int(pd.Timestamp(C.SIM_START).timestamp())
DAY = 86400
HORIZON = C.SIM_DAYS * DAY
TRAIN_END = C.TRAIN_DAYS * DAY

CITIES = np.array(["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune",
                   "Kolkata", "Ahmedabad", "Jaipur", "Coimbatore"], dtype=object)
CITY_P = np.array([.16, .15, .14, .12, .10, .08, .07, .07, .05, .06])
CITY_P = CITY_P / CITY_P.sum()
HOUR_W = np.array([1, .5, .3, .3, .4, 1, 2, 4, 6, 7, 7, 7, 7, 6, 6, 6, 7, 8, 9, 9, 8, 6, 4, 2], float)
HOUR_P = HOUR_W / HOUR_W.sum()

P2P_REM = ["", "", "", "dinner split", "thanks", "lunch", "gift", "movie", "groceries", "chai",
           "cab share", "loan return", "birthday", "for you", "rent share", "trip"]
MERCH_REM = ["", "", "", "", "payment", "order", "food", "groceries", "shopping", "recharge", "subscription"]
BILL_REM = ["electricity bill", "mobile recharge", "broadband", "water bill", "insurance premium",
            "gas cylinder", "dth recharge"]
BIZ_REM = ["invoice", "supplier payment", "stock purchase", "inv", "payment", ""]
SCAM_REM = ["urgent", "KYC update", "refund", "prize claim", "processing fee", "account verification",
            "urgent help", "reward", "refund claim", "verify account"]
# probability that a fraud txn carries a scam-style remark (else an ordinary one)
SCAM_P = {"account_takeover": .45, "velocity_abuse": .20, "mule_passthrough": .55,
          "circular": .05, "fan_in": .50, "fan_out": .20, "fragmentation": .12}
SCAM_P_NORMAL = (.012, .003, .001)  # normal txns that say refund / urgent / KYC update (overlap)

BASE_COUNTS = {"fragmentation": 11, "account_takeover": 36, "velocity_abuse": 14,
               "mule_passthrough": 28, "circular": 18, "fan_in": 10, "fan_out": 10}


# ----------------------------------------------------------------- buffer
class Buf:
    def __init__(self):
        self.parts, self.n = [], 0

    def add(self, ts, src, dst, amount, channel, origin_type, device, ip, city, remark,
            label=0, typology="none", scenario=""):
        ts = np.atleast_1d(np.asarray(ts, dtype=np.int64))
        m = len(ts)
        if m == 0:
            return
        def b(x, dt=object):
            if np.isscalar(x):
                return np.full(m, x, dtype=dt)
            a = np.asarray(x, dtype=dt)
            assert len(a) == m, (len(a), m)
            return a
        part = pd.DataFrame({
            "ts": ts, "src": b(src), "dst": b(dst), "amount": b(amount, float),
            "channel": b(channel), "origin_type": b(origin_type), "device_id": b(device),
            "ip": b(ip), "city": b(city), "remark": b(remark),
            "label": np.full(m, label, dtype=np.int8), "typology": b(typology),
            "scenario_id": b(scenario), "rid": np.arange(self.n, self.n + m),
        })
        self.n += m
        self.parts.append(part)


class Generator:
    def __init__(self, n_accounts=C.N_ACCOUNTS, seed=C.SEED):
        self.n = n_accounts
        self.rng = np.random.default_rng(seed)
        self.scale = n_accounts / 5000.0
        self.buf = Buf()
        self.meta = {}   # scenario_id -> total_amount
        self.plans = []

    # -------------------------------------------------------------- helpers
    def A(self, x):
        return self.ids[x]

    def take(self, pool, k):
        k = int(k)
        out = [pool.pop() for _ in range(k)]
        return np.array(out, dtype=np.int64)

    def remark_normal(self, kind, m, overlap=True):
        rng = self.rng
        opts = {"p2p": P2P_REM, "merch": MERCH_REM, "bill": BILL_REM, "biz": BIZ_REM}[kind]
        r = rng.choice(np.array(opts, dtype=object), m)
        if overlap:
            u = rng.random(m)
            a, b_, c_ = SCAM_P_NORMAL
            r = np.where(u < a, "refund", r)
            r = np.where((u >= a) & (u < a + b_), "urgent", r)
            r = np.where((u >= a + b_) & (u < a + b_ + c_), "KYC update", r)
        return np.asarray(r, dtype=object)

    def remark_fraud(self, typ, m):
        rng = self.rng
        scam = rng.random(m) < SCAM_P[typ]
        r = np.where(scam, rng.choice(np.array(SCAM_REM, dtype=object), m),
                     rng.choice(np.array(P2P_REM, dtype=object), m))
        return np.asarray(r, dtype=object)

    def rand_ts(self, m):
        rng = self.rng
        day = rng.integers(0, C.SIM_DAYS, m)
        hour = rng.choice(24, m, p=HOUR_P)
        return day * DAY + hour * 3600 + rng.integers(0, 3600, m)

    def ctx(self, actor, ts):
        """device / ip / city for txns initiated by `actor` (account idx array)."""
        rng = self.rng
        actor = np.asarray(actor, dtype=np.int64)
        ts = np.asarray(ts)
        m = len(actor)
        dev = np.where(ts >= self.sw_ts[actor], self.new_dev[actor], self.prim_dev[actor])
        use_sec = (rng.random(m) < 0.04) & self.has_sec[actor]
        dev = np.where(use_sec, self.sec_dev[actor], dev)
        intravel = (ts >= self.tr_s[actor]) & (ts < self.tr_e[actor])
        city = self.city_names[np.where(intravel, self.tr_city[actor], self.city[actor])]
        octet = rng.integers(2, 255, m).astype(str)
        ip = np.char.add(self.ip_pref[actor].astype(str), np.char.add(".", octet)).astype(object)
        return np.asarray(dev, dtype=object), ip, np.asarray(city, dtype=object)

    def nrm(self, src, dst, amt, ts, channel, origin, remark, actor):
        """Emit legitimate traffic (label 0)."""
        src = np.asarray(src, dtype=object)
        dst = np.asarray(dst, dtype=object)
        keep = src != dst
        amt = np.asarray(amt, dtype=float)
        keep &= amt > 0
        ts = np.minimum(np.asarray(ts, dtype=np.int64), HORIZON - 1)
        sel = np.where(keep)[0]
        pick = lambda x: x if np.isscalar(x) else np.asarray(x)[sel]
        ts, actor = ts[sel], np.asarray(actor)[sel]
        dev, ip, city = self.ctx(actor, ts)
        self.buf.add(ts, src[sel], dst[sel], amt[sel], pick(channel), pick(origin),
                     dev, ip, city, pick(remark))

    def sc(self, sid, typ, ts, src, dst, amt, channel, origin, actor, remark,
           device=None, ip=None, city=None):
        """Emit fraud txns (label 1)."""
        ts = np.minimum(np.atleast_1d(np.asarray(ts, dtype=np.int64)), HORIZON - 1)
        m = len(ts)
        actor = np.full(m, actor, dtype=np.int64) if np.isscalar(actor) else np.asarray(actor)
        dev, ipp, cty = self.ctx(actor, ts)
        if device is not None:
            dev = device
        if ip is not None:
            ipp = ip
        if city is not None:
            cty = city
        self.buf.add(ts, src, dst, amt, channel, origin, dev, ipp, cty, remark,
                     label=1, typology=typ, scenario=sid)

    # ------------------------------------------------------------- accounts
    def build_accounts(self):
        rng, n = self.rng, self.n
        self.ids = np.array([f"ACC{i + 1:05d}" for i in range(n)], dtype=object)
        perm = rng.permutation(n)
        n_m, n_b = int(round(.12 * n)), int(round(.08 * n))
        atype = np.array(["individual"] * n, dtype=object)
        atype[perm[:n_m]] = "merchant"
        atype[perm[n_m:n_m + n_b]] = "business"
        self.atype = atype
        ind = np.where(atype == "individual")[0]
        self.merch_idx = np.where(atype == "merchant")[0]
        self.bus_idx = np.where(atype == "business")[0]

        kyc = np.where(atype == "individual", rng.choice([1, 2, 3], n, p=[.3, .5, .2]),
                       rng.choice([2, 3], n))
        created = T0 - rng.integers(60, 2500, n) * DAY - rng.integers(0, DAY, n)
        self.city = rng.choice(len(CITIES), n, p=CITY_P)
        self.city_names = CITIES
        self.prim_dev = np.array([f"DEV{i + 1:06d}" for i in range(n)], dtype=object)
        self.sec_dev = np.array([f"DEV{n + i + 1:06d}" for i in range(n)], dtype=object)
        self.new_dev = np.array([f"DEV{2 * n + i + 1:06d}" for i in range(n)], dtype=object)
        self.has_sec = rng.random(n) < .25
        pref_pool = np.empty((len(CITIES), 40), dtype=object)
        firsts = [49, 106, 117, 157, 171, 182, 223, 103]
        for c in range(len(CITIES)):
            for k in range(40):
                pref_pool[c, k] = f"{firsts[(c + k) % 8]}.{rng.integers(0, 255)}.{rng.integers(0, 255)}"
        self.ip_pref = pref_pool[self.city, rng.integers(0, 40, n)]
        self.role = np.array(["normal"] * n, dtype=object)

        # dormant mule pool (no organic traffic)
        n_mule = max(40, int(round(150 * self.scale)))
        mule_idx = rng.choice(ind, n_mule, replace=False)
        self.dormant = np.zeros(n, bool)
        self.dormant[mule_idx] = True
        kyc[mule_idx] = 1
        created[mule_idx] = T0 - rng.integers(200, 900, n_mule) * DAY
        self.act_idx = np.array([i for i in ind if not self.dormant[i]], dtype=np.int64)
        # organised mule groups share device and/or ip prefix
        ml = list(mule_idx)
        while ml:
            g = [ml.pop() for _ in range(min(len(ml), int(rng.integers(3, 6))))]
            if rng.random() < .6:
                self.prim_dev[g] = self.prim_dev[g[0]]
            if rng.random() < .5:
                self.ip_pref[g] = self.ip_pref[g[0]]
        self.mule_pool = list(rng.permutation(mule_idx))
        self.act_pool = list(rng.permutation(self.act_idx))

        # legit hard negative: families sharing a device / ip prefix
        for _ in range(int(round(120 * self.scale))):
            g = rng.choice(self.act_idx, int(rng.integers(2, 4)), replace=False)
            self.prim_dev[g] = self.prim_dev[g[0]]
            self.ip_pref[g] = self.ip_pref[g[0]]
            self.city[g] = self.city[g[0]]

        # legit device switch (new phone) and travel windows
        self.sw_ts = np.full(n, np.inf)
        sw = rng.choice(self.act_idx, int(.06 * len(self.act_idx)), replace=False)
        self.sw_ts[sw] = rng.integers(2 * DAY, 28 * DAY, len(sw))
        self.tr_s = np.full(n, np.inf)
        self.tr_e = np.full(n, np.inf)
        self.tr_city = self.city.copy()
        tr = rng.choice(self.act_idx, int(.10 * len(self.act_idx)), replace=False)
        self.tr_s[tr] = rng.integers(1 * DAY, 27 * DAY, len(tr))
        self.tr_e[tr] = self.tr_s[tr] + rng.integers(1, 4, len(tr)) * DAY
        self.tr_city[tr] = (self.city[tr] + rng.integers(1, len(CITIES), len(tr))) % len(CITIES)

        self.kyc, self.created = kyc, created
        self.phone = np.array([hashlib.sha1(f"phone{i}".encode()).hexdigest()[:12] for i in range(n)],
                              dtype=object)

    # ------------------------------------------------------------- planning
    def starts(self, k):
        rng = self.rng
        n_test = max(2, int(round(k * .3)))
        n_train = k - n_test
        tr = np.linspace(.3, 19.6, n_train) + rng.uniform(-.2, .2, n_train)
        te = np.linspace(21.3, 28.2, n_test) + rng.uniform(-.2, .2, n_test)
        d = np.clip(np.concatenate([tr, te]), .25, 28.4)
        d = np.where(np.arange(k) < n_train, np.minimum(d, 19.8), np.maximum(d, 21.2))
        return (d * DAY).astype(np.int64)

    def count(self, typ):
        return max(4, int(round(BASE_COUNTS[typ] * self.scale)))

    def plan_all(self):
        rng = self.rng
        ap, mp = self.act_pool, self.mule_pool
        shr = 0
        for t0 in self.starts(self.count("fragmentation")):
            R, L = int(rng.integers(15, 41)), int(rng.integers(2, 5))
            relays = self.take(ap, R)
            layers = np.array_split(relays, L)
            coll = self.take(ap, rng.integers(2, 4))
            org = self.take(ap, 1)[0]
            self.role[relays] = "relay"
            self.role[coll] = "collector"
            self.role[org] = "origin"
            D = rng.uniform(200_000, 1_000_000)
            n_lo, n_hi = max(40, int(np.ceil(D / 9900))), min(150, int(D // 3000))
            n = int(rng.integers(n_lo, n_hi + 1))
            amts = np.clip((D / n) * rng.uniform(.8, 1.2, n), 3000, 9900).astype(np.int64)
            mode = rng.choice(["none", "device", "ip", "both"], p=[.55, .15, .15, .15])
            if mode != "none":
                sub = rng.choice(relays, max(3, int(.5 * R)), replace=False)
                if mode in ("device", "both"):
                    self.prim_dev[sub] = f"DEV_SHR{shr:03d}"
                if mode in ("ip", "both"):
                    self.ip_pref[sub] = f"152.{58 + shr}.{rng.integers(0, 255)}"
                shr += 1
            self.plans.append(("fragmentation", dict(
                start=int(t0), origin=org, layers=layers, collectors=coll, n=n, amts=amts,
                dwell=float(rng.choice([4, 12, 30, 75, 150])))))
        for t0 in self.starts(self.count("account_takeover")):
            self.plans.append(("account_takeover", dict(
                start=int(t0), victim=self.take(ap, 1)[0], mule=self.take(mp, 1)[0])))
            self.role[self.plans[-1][1]["victim"]] = "ato_victim"
            self.role[self.plans[-1][1]["mule"]] = "mule"
        for t0 in self.starts(self.count("velocity_abuse")):
            a = self.take(mp, 1)[0]
            self.role[a] = "mule"
            self.plans.append(("velocity_abuse", dict(start=int(t0), acct=a)))
        for t0 in self.starts(self.count("mule_passthrough")):
            m = self.take(mp, 2)
            self.role[m] = "mule"
            self.plans.append(("mule_passthrough", dict(
                start=int(t0), victim=self.take(ap, 1)[0], m1=m[0], m2=m[1])))
        for t0 in self.starts(self.count("circular")):
            nodes = self.take(ap, rng.integers(3, 6))
            self.role[nodes] = "relay"
            if rng.random() < .3:
                self.prim_dev[nodes] = self.prim_dev[nodes[0]]
            self.plans.append(("circular", dict(start=int(t0), nodes=nodes)))
        for t0 in self.starts(self.count("fan_in")):
            hub = self.take(mp, 1)[0]
            self.role[hub] = "mule"
            m = int(rng.integers(20, 46))
            senders = rng.choice(self.act_idx, m, replace=False)
            self.plans.append(("fan_in", dict(start=int(t0), hub=hub, senders=senders)))
        for t0 in self.starts(self.count("fan_out")):
            hub = self.take(mp, 1)[0]
            self.role[hub] = "mule"
            D = int(rng.integers(100_000, 250_000) // 1000 * 1000)
            m = int(np.clip(np.ceil(D / rng.integers(5000, 9000)), 20, 60))
            recv = self.take(ap, m)
            self.role[recv] = "relay"
            self.plans.append(("fan_out", dict(start=int(t0), hub=hub, D=D, recv=recv)))

    # ------------------------------------------------------- normal traffic
    def normal_traffic(self):
        rng, n, ids = self.rng, self.n, self.ids
        act, mer, bus = self.act_idx, self.merch_idx, self.bus_idx
        mult = rng.gamma(2.0, .5, n)
        billers, shop = mer[:20], mer[20:]
        pop = 1.0 / (np.arange(len(shop)) + 1) ** .9
        pop /= pop.sum()
        self.shop = shop
        contacts = rng.choice(act, size=(n, 10))
        favs = rng.choice(shop, size=(n, 8), p=pop)
        skew = lambda k, m: np.minimum(rng.integers(0, k, m), rng.integers(0, k, m))
        chan = lambda m: np.where(rng.random(m) < .88, "UPI", "IMPS").astype(object)

        # P2P
        src = np.repeat(act, rng.poisson(34 * mult[act]))
        m = len(src)
        dst = contacts[src, skew(10, m)]
        r = rng.random(m) < .10
        dst[r] = rng.choice(act, r.sum())
        amt = np.round(np.clip(rng.lognormal(6.4, 1.15, m), 10, 95_000))
        self.nrm(ids[src], ids[dst], amt, self.rand_ts(m), chan(m), "TRANSFER",
                 self.remark_normal("p2p", m), src)

        # merchant payments
        src = np.repeat(act, rng.poisson(70 * mult[act]))
        m = len(src)
        dst = favs[src, skew(8, m)]
        r = rng.random(m) < .12
        dst[r] = rng.choice(shop, r.sum())
        big = rng.random(m) < .05
        amt = np.where(big, np.clip(rng.lognormal(8.6, .9, m), 500, 90_000),
                       np.clip(rng.lognormal(5.9, .95, m), 5, 20_000))
        amt = np.where(rng.random(m) < .5, np.round(amt), np.round(amt, 2))
        self.nrm(ids[src], ids[dst], amt, self.rand_ts(m), "UPI", "TRANSFER",
                 self.remark_normal("merch", m), src)

        # utility bills (fixed billers, fixed due day)
        bill_dst = rng.choice(billers, size=(n, 4))
        bill_amt = np.clip(rng.lognormal(6.6, .7, (n, 4)), 150, 9000)
        bill_day = rng.integers(0, 28, (n, 4))
        acc = np.repeat(act, 4)
        b = np.tile(np.arange(4), len(act))
        k = rng.random(len(acc)) < .7
        acc, b = acc[k], b[k]
        m = len(acc)
        amt = np.round(bill_amt[acc, b] * rng.uniform(.97, 1.03, m), 0)
        ts = bill_day[acc, b] * DAY + rng.integers(8 * 3600, 22 * 3600, m)
        self.nrm(ids[acc], ids[bill_dst[acc, b]], amt, ts, "UPI", "TRANSFER",
                 np.array(BILL_REM, dtype=object)[b % len(BILL_REM)], acc)

        # rent
        renters = rng.choice(act, int(.35 * len(act)), replace=False)
        land = rng.choice(act, len(renters))
        m = len(renters)
        amt = rng.choice(np.arange(6000, 36000, 500), m).astype(float)
        ts = rng.integers(0, 29, m) * DAY + rng.integers(8 * 3600, 22 * 3600, m)
        self.nrm(ids[renters], ids[land], amt, ts, "UPI", "TRANSFER",
                 rng.choice(np.array(["rent", "house rent", "rent jan", "rent"], dtype=object), m), renters)

        # salary credits
        earn = rng.choice(act, int(.7 * len(act)), replace=False)
        m = len(earn)
        amt = np.round(np.clip(rng.lognormal(10.6, .5, m), 12_000, 250_000), -2)
        ts = rng.integers(0, 30, m) * DAY + rng.integers(9 * 3600, 12 * 3600, m)
        self.nrm(np.full(m, "EXT_EMPLOYER", dtype=object), ids[earn], amt, ts, "NEFT", "SALARY",
                 rng.choice(np.array(["salary", "salary jan", "sal credit", ""], dtype=object), m), earn)

        # cash deposits (individuals small, merchants larger and frequent) and withdrawals
        for pool, lam, mu, sg, cap in ((rng.choice(act, int(.05 * len(act)), replace=False), 1.5, 9.0, .8, 40_000),
                                       (mer, 5.0, 10.3, .7, 300_000)):
            s = np.repeat(pool, rng.poisson(lam, len(pool)))
            m = len(s)
            a = np.round(np.clip(rng.lognormal(mu, sg, m), 1000, cap), -2)
            self.nrm(np.full(m, "EXT_CASH", dtype=object), ids[s], a, self.rand_ts(m), "CASH_DEP", "CASH",
                     rng.choice(np.array(["", "", "cash deposit"], dtype=object), m), s)
        wd = np.repeat(rng.choice(act, int(.25 * len(act)), replace=False), 0)
        wd = rng.choice(act, int(.25 * len(act)), replace=False)
        s = np.repeat(wd, rng.poisson(2.0, len(wd)))
        m = len(s)
        self.nrm(ids[s], np.full(m, "EXT_CASH_OUT", dtype=object), rng.choice(np.arange(500, 10_001, 500), m).astype(float),
                 self.rand_ts(m), "CASH_WD", "TRANSFER", np.full(m, "atm", dtype=object), s)

        # loans, unknown-origin credits
        la = rng.choice(act, max(1, int(.015 * len(act))), replace=False)
        m = len(la)
        self.nrm(np.full(m, "EXT_LOAN", dtype=object), ids[la],
                 rng.choice([50_000, 100_000, 200_000, 300_000, 500_000], m).astype(float),
                 rng.integers(0, 30, m) * DAY + 11 * 3600, "NEFT", "LOAN",
                 np.full(m, "loan disbursal", dtype=object), la)
        ua = rng.choice(act, max(1, int(.01 * len(act))), replace=False)
        m = len(ua)
        self.nrm(np.full(m, "EXT_UNKNOWN", dtype=object), ids[ua], rng.integers(300, 15_000, m).astype(float),
                 self.rand_ts(m), "UPI", "UNKNOWN",
                 rng.choice(np.array(["refund", "cashback", ""], dtype=object), m), ua)

        # business / merchant B2B traffic (some legitimately >= Rs 50k)
        payers = np.concatenate([bus, mer])
        lam = np.where(np.isin(payers, bus), 20, 6) * mult[payers]
        sup = rng.choice(bus, size=(n, 6))
        s = np.repeat(payers, rng.poisson(lam))
        m = len(s)
        d = sup[s, skew(6, m)]
        a = np.round(np.clip(rng.lognormal(9.3, 1.1, m), 800, 800_000), 0)
        ch = np.where(a > 100_000, rng.choice(["IMPS", "NEFT"], m),
                      rng.choice(["UPI", "IMPS", "NEFT"], m, p=[.4, .35, .25])).astype(object)
        self.nrm(ids[s], ids[d], a, self.rand_ts(m), ch, "TRANSFER", self.remark_normal("biz", m, False), s)
        s = np.repeat(bus, rng.poisson(5, len(bus)))
        m = len(s)
        a = np.round(np.clip(rng.lognormal(10.2, 1.0, m), 2000, 900_000), 0)
        self.nrm(np.full(m, "EXT_BUSINESS", dtype=object), ids[s], a, self.rand_ts(m),
                 rng.choice(np.array(["NEFT", "IMPS"], dtype=object), m), "BUSINESS",
                 rng.choice(np.array(["client payment", "invoice"], dtype=object), m), s)

        self.hard_negatives(act, bus, mer, shop)

    def hard_negatives(self, act, bus, mer, shop):
        rng, ids, sc = self.rng, self.ids, self.scale
        ev = lambda base, lo=5: max(lo, int(round(base * sc)))

        # legit fan-in: festival rush at a shop (many small payments from different customers)
        for _ in range(ev(15)):
            mm = int(rng.integers(80, 200))
            s = rng.choice(act, mm)
            ts = int(rng.integers(0, 29)) * DAY + rng.integers(10 * 3600, 16 * 3600, mm)
            self.nrm(ids[s], np.full(mm, ids[rng.choice(shop)], dtype=object),
                     np.round(np.clip(rng.lognormal(7.0, .7, mm), 100, 5000)), ts, "UPI", "TRANSFER",
                     rng.choice(np.array(["pongal sale", "republic day sale", "payment", "order", "sale"], dtype=object), mm), s)
        # friends splitting a bill
        for _ in range(ev(500, 20)):
            g = rng.choice(act, int(rng.integers(4, 7)), replace=False)
            total = int(rng.integers(1500, 9000))
            t = int(rng.integers(0, 30)) * DAY + int(rng.integers(12 * 3600, 23 * 3600))
            rest = ids[rng.choice(shop)]
            self.nrm([ids[g[0]]], [rest], [float(total)], [t], "UPI", "TRANSFER", ["dinner"], [g[0]])
            k = len(g) - 1
            self.nrm(ids[g[1:]], np.full(k, ids[g[0]], dtype=object), np.full(k, float(round(total / (k + 1)))),
                     t + rng.integers(300, 5400, k), "UPI", "TRANSFER",
                     rng.choice(np.array(["dinner split", "my share", "split", "lunch split"], dtype=object), k), g[1:])
        # freelancers receiving several mid-size payments (some above threshold)
        for _ in range(ev(40, 4)):
            f = rng.choice(act)
            k = int(rng.integers(3, 7))
            clients = rng.choice(np.concatenate([bus, act]), k)
            a = np.where(rng.random(k) < .15, rng.integers(55, 90, k) * 1000, rng.integers(24, 97, k) * 500).astype(float)
            self.nrm(ids[clients], np.full(k, ids[f], dtype=object), a, rng.integers(0, 30 * DAY, k),
                     rng.choice(np.array(["UPI", "IMPS", "NEFT"], dtype=object), k), "TRANSFER",
                     rng.choice(np.array(["invoice", "project payment", "design work", "consulting", "freelance work"], dtype=object), k),
                     clients)
        # weddings: large advance plus follow-up payments
        for _ in range(ev(60)):
            p, v = rng.choice(act), shop[rng.integers(0, len(shop))]
            t = int(rng.integers(0, 24)) * DAY + int(rng.integers(9 * 3600, 20 * 3600))
            self.nrm([ids[p]], [ids[v]], [float(rng.integers(20, 80) * 5000)], [t],
                     rng.choice(["IMPS", "NEFT"]), "TRANSFER", [rng.choice(["wedding advance", "venue booking", "catering advance"])], [p])
            k = int(rng.integers(3, 8))
            vs = rng.choice(shop, k)
            self.nrm(np.full(k, ids[p], dtype=object), ids[vs], rng.integers(30, 180, k) * 500.0,
                     np.minimum(t + rng.integers(3600, 6 * DAY, k), HORIZON - 1), "IMPS", "TRANSFER",
                     rng.choice(np.array(["jewellery", "decor", "wedding shopping", "photographer", "catering balance"], dtype=object), k),
                     np.full(k, p))
        # tuition / college fees, includes round amounts just under the alert threshold
        colleges = shop[:6]
        for _ in range(ev(80)):
            p = rng.choice(act)
            a = float(rng.choice([25_000, 30_000, 40_000, 45_000, 48_000, 49_000, 50_000, 60_000, 75_000, 85_000, 120_000, 150_000]))
            self.nrm([ids[p]], [ids[rng.choice(colleges)]], [a], [int(rng.integers(0, 30)) * DAY + int(rng.integers(9 * 3600, 20 * 3600))],
                     "IMPS" if a > 100_000 else rng.choice(["UPI", "IMPS", "NEFT"]), "TRANSFER",
                     [rng.choice(["tuition fee", "semester fee", "college fee", "fees"])], [p])
        # legit new phone + travel + big purchase (looks like ATO, is not)
        for _ in range(ev(150)):
            p = rng.choice(act)
            t = int(rng.integers(1, 29)) * DAY + int(rng.integers(8 * 3600, 22 * 3600))
            kk = int(rng.integers(0, 3))
            ts = t + np.concatenate([np.cumsum(rng.integers(60, 600, kk)), [kk * 600 + 900]]).astype(np.int64)
            a = np.concatenate([rng.integers(200, 3000, kk), [rng.integers(30, 95) * 1000]]).astype(float)
            dest = rng.choice(shop, len(ts))
            c = (self.city[p] + int(rng.integers(1, len(CITIES)))) % len(CITIES)
            octet = rng.integers(2, 255, len(ts)).astype(str)
            self.buf.add(ts, np.full(len(ts), ids[p], dtype=object), ids[dest], a, "UPI", "TRANSFER",
                         self.new_dev[p], np.char.add(self.ip_pref[p] + ".", octet).astype(object), CITIES[c],
                         rng.choice(np.array(["hotel booking", "laptop", "phone purchase", "flight tickets", "jewellery", "holiday package"], dtype=object), len(ts)))

    # ------------------------------------------------------------- emitters
    def cashout(self, sid, typ, who, tot, t_start):
        """structured cash withdrawals below the alert threshold."""
        rng = self.rng
        remaining, t = int(tot * rng.uniform(.97, .995)), int(t_start)
        ts, amts = [], []
        while remaining > 0:
            c = int(min(remaining, rng.integers(20_000, 49_500) // 500 * 500 or 500))
            t += int(rng.integers(180, 2400))
            ts.append(t)
            amts.append(c)
            remaining -= c
        m = len(ts)
        self.sc(sid, typ, ts, np.full(m, self.ids[who], dtype=object), np.full(m, "EXT_CASH_OUT", dtype=object),
                np.array(amts, float), "CASH_WD", "TRANSFER", np.full(m, who),
                np.full(m, "", dtype=object))

    def emit_frag(self, sid, p):
        rng, A = self.rng, self.A
        n, t0, org = p["n"], p["start"], p["origin"]
        amts = p["amts"].astype(np.int64)
        D = int(amts.sum())
        self.meta[sid] = D
        self.sc(sid, "fragmentation", [t0], [ "EXT_CASH"], [A(org)], [float(D)], "CASH_DEP", "CASH", org,
                self.remark_fraud("fragmentation", 1))
        chain = [np.full(n, org)] + [rng.choice(l, n) for l in p["layers"]]
        coll = rng.choice(p["collectors"], n)
        chain.append(coll)
        cur_t = t0 + 180 + rng.integers(0, int(rng.integers(900, 10_800)), n)
        cur_a = amts.copy()
        for h in range(len(chain) - 1):
            if h > 0:
                dwell = np.clip(rng.lognormal(np.log(p["dwell"] * 60), .9, n), 90, 3 * 3600).astype(np.int64)
                cur_t = cur_t + dwell
                cur_a = np.floor(cur_a * (1 - rng.uniform(0, .004, n))).astype(np.int64)
            s, d = chain[h], chain[h + 1]
            self.sc(sid, "fragmentation", cur_t, A(s), A(d), cur_a.astype(float),
                    np.where(rng.random(n) < .8, "UPI", "IMPS").astype(object), "TRANSFER", s,
                    self.remark_fraud("fragmentation", n))
        for c in p["collectors"]:
            msk = coll == c
            if msk.any():
                self.cashout(sid, "fragmentation", c, cur_a[msk].sum(), cur_t[msk].max() + rng.integers(600, 7200))

    def emit_mule(self, sid, p):
        rng, A = self.rng, self.A
        v, m1, m2, t = p["victim"], p["m1"], p["m2"], p["start"]
        a = int(rng.integers(15_000, 98_000)) if rng.random() < .8 else int(rng.integers(100_000, 180_000))
        a = a // 100 * 100
        self.meta[sid] = a
        t2 = t + int(rng.integers(60, 900))
        a2 = int(a * (1 - rng.uniform(0, .01)))
        t3 = t2 + int(rng.integers(120, 1200))
        a3 = int(a2 * (1 - rng.uniform(0, .01)))
        ch = lambda x: "UPI" if x <= 100_000 else "IMPS"
        self.sc(sid, "mule_passthrough", [t], [A(v)], [A(m1)], [float(a)], ch(a), "TRANSFER", v, self.remark_fraud("mule_passthrough", 1))
        self.sc(sid, "mule_passthrough", [t2], [A(m1)], [A(m2)], [float(a2)], ch(a2), "TRANSFER", m1, self.remark_fraud("mule_passthrough", 1))
        self.sc(sid, "mule_passthrough", [t3], [A(m2)], ["EXT_CASH_OUT"], [float(a3)], "CASH_WD", "TRANSFER", m2, [""])

    def emit_ato(self, sid, p):
        rng, A = self.rng, self.A
        v, mu, t = p["victim"], p["mule"], p["start"]
        atk = f"DEV_ATK{int(rng.integers(0, 6 if rng.random() < .3 else 10_000)):04d}"
        pref = f"103.{int(rng.integers(240, 256))}.{int(rng.integers(0, 255))}"
        city = CITIES[(self.city[v] + int(rng.integers(1, len(CITIES)))) % len(CITIES)]
        kp = int(rng.integers(0, 3))
        ts = [t + int(x) for x in np.cumsum(rng.integers(20, 300, kp))] if kp else []
        am = [float(x) for x in rng.integers(1, 501, kp)]
        tb = (ts[-1] if ts else t) + int(rng.integers(60, 600))
        a1 = int(rng.integers(28_000, 99_000)) if rng.random() < .8 else int(rng.integers(100_000, 250_000))
        ts.append(tb)
        am.append(float(a1))
        big = a1
        if rng.random() < .5:
            a2 = int(rng.integers(8_000, 60_000))
            ts.append(tb + int(rng.integers(300, 2400)))
            am.append(float(a2))
            big += a2
        self.meta[sid] = big
        k = len(ts)
        ip = np.char.add(pref + ".", rng.integers(2, 255, k).astype(str)).astype(object)
        self.sc(sid, "account_takeover", ts, np.full(k, A(v), dtype=object), np.full(k, A(mu), dtype=object), am,
                np.where(np.array(am) > 100_000, "IMPS", "UPI").astype(object), "TRANSFER", v,
                self.remark_fraud("account_takeover", k), device=atk, ip=ip, city=city)
        self.sc(sid, "account_takeover", [max(ts) + int(rng.integers(180, 1800))], [A(mu)], ["EXT_CASH_OUT"],
                [float(int(big * .97))], "CASH_WD", "TRANSFER", mu, [""])

    def emit_velocity(self, sid, p):
        rng, A = self.rng, self.A
        a, t = p["acct"], p["start"]
        N = int(rng.integers(25, 61))
        dur = int(rng.integers(20 * 60, 90 * 60))
        ts = t + np.sort(rng.integers(0, dur, N))
        amt = np.where(rng.random(N) < .3, rng.choice([499, 999, 1999, 2999, 4999], N), rng.integers(100, 4900, N)).astype(float)
        tot = float(amt.sum())
        self.meta[sid] = tot
        self.sc(sid, "velocity_abuse", [t - int(rng.integers(300, 1800))], ["EXT_UNKNOWN"], [A(a)], [tot + 500], "IMPS", "UNKNOWN", a, [""])
        d = rng.choice(np.concatenate([self.act_idx, self.shop]), N)
        d = np.where(d == a, self.act_idx[0], d)
        bot = np.where(rng.random(N) < .6, f"DEV_BOT{int(rng.integers(0, 40)):03d}", self.prim_dev[a]).astype(object)
        dev, ip, city = self.ctx(np.full(N, a), ts)
        self.sc(sid, "velocity_abuse", ts, np.full(N, A(a), dtype=object), A(d), amt, "UPI", "TRANSFER", a,
                self.remark_fraud("velocity_abuse", N), device=bot)

    def emit_circular(self, sid, p):
        rng, A = self.rng, self.A
        nodes, t = p["nodes"], p["start"]
        k = len(nodes)
        amt = float(int(rng.integers(8_000, 46_000)) // 100 * 100)
        self.meta[sid] = amt
        ts, s, d, am = [], [], [], []
        for _ in range(int(rng.integers(2, 4))):
            for j in range(k):
                t += int(rng.integers(120, 5400))
                amt = float(int(amt * (1 - rng.uniform(0, .005))))
                ts.append(t); s.append(nodes[j]); d.append(nodes[(j + 1) % k]); am.append(amt)
        s, d = np.array(s), np.array(d)
        self.sc(sid, "circular", ts, A(s), A(d), am, "UPI", "TRANSFER", s, self.remark_fraud("circular", len(ts)))

    def emit_fan_in(self, sid, p):
        rng, A = self.rng, self.A
        hub, snd, t = p["hub"], p["senders"], p["start"]
        m = len(snd)
        ts = t + np.sort(rng.integers(0, int(rng.integers(1800, 10_800)), m))
        am = (rng.integers(20, 190, m) * 50).astype(float)
        self.meta[sid] = float(am.sum())
        self.sc(sid, "fan_in", ts, A(snd), np.full(m, A(hub), dtype=object), am,
                np.where(rng.random(m) < .85, "UPI", "IMPS").astype(object), "TRANSFER", snd, self.remark_fraud("fan_in", m))
        self.cashout(sid, "fan_in", hub, am.sum(), ts.max() + rng.integers(600, 3600))

    def emit_fan_out(self, sid, p):
        rng, A = self.rng, self.A
        hub, D, recv, t = p["hub"], p["D"], p["recv"], p["start"]
        m = len(recv)
        self.meta[sid] = float(D)
        self.sc(sid, "fan_out", [t], ["EXT_CASH"], [A(hub)], [float(D)], "CASH_DEP", "CASH", hub, [""])
        am = np.clip((D / m) * rng.uniform(.8, 1.2, m), 2000, 9900).astype(np.int64).astype(float)
        ts = t + int(rng.integers(600, 3600)) + np.sort(rng.integers(0, int(rng.integers(1800, 7200)), m))
        self.sc(sid, "fan_out", ts, np.full(m, A(hub), dtype=object), A(recv), am, "UPI", "TRANSFER", hub,
                self.remark_fraud("fan_out", m))

    def emit_all(self):
        order = sorted(range(len(self.plans)), key=lambda i: (self.plans[i][1]["start"], i))
        fn = {"fragmentation": self.emit_frag, "account_takeover": self.emit_ato,
              "velocity_abuse": self.emit_velocity, "mule_passthrough": self.emit_mule,
              "circular": self.emit_circular, "fan_in": self.emit_fan_in, "fan_out": self.emit_fan_out}
        for k, i in enumerate(order, start=1):
            typ, p = self.plans[i]
            fn[typ](f"S{k:04d}", p)

    # ------------------------------------------------------------- finalize
    def finalize(self):
        df = pd.concat(self.buf.parts, ignore_index=True)
        order = np.lexsort((df["rid"].values, df["ts"].values))
        df = df.iloc[order].reset_index(drop=True)
        df["txn_id"] = ["T%07d" % (i + 1) for i in range(len(df))]
        first = df.loc[df["scenario_id"] != ""].groupby("scenario_id")["txn_id"].first()
        launder = df["typology"].isin(S.LAUNDERING_TYPOLOGIES)
        df["root_txn_id_gt"] = np.where(launder, df["scenario_id"].map(first).fillna(""), "")
        df["ts_abs"] = df["ts"] + T0
        iso = pd.to_datetime(df["ts_abs"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        txns = pd.DataFrame({
            "txn_id": df["txn_id"], "ts": iso, "src_account": df["src"], "dst_account": df["dst"],
            "amount": df["amount"].round(2), "channel": df["channel"], "origin_type": df["origin_type"],
            "device_id": df["device_id"], "ip": df["ip"], "city": df["city"], "remark": df["remark"],
            "label": df["label"].astype(int), "typology": df["typology"], "scenario_id": df["scenario_id"],
            "root_txn_id_gt": df["root_txn_id_gt"]})[S.TXN_COLUMNS]
        n = self.n
        accounts = pd.DataFrame({
            "account_id": self.ids,
            "created_at": pd.to_datetime(self.created, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kyc_tier": self.kyc, "account_type": self.atype, "city": self.city_names[self.city],
            "primary_device_id": self.prim_dev, "primary_ip_prefix": self.ip_pref,
            "phone_hash": self.phone, "role_gt": self.role})[S.ACCOUNT_COLUMNS]
        pos = txns[txns["label"] == 1]
        rows = []
        for sid, g in pos.groupby("scenario_id", sort=True):
            accs = (set(g["src_account"]) | set(g["dst_account"])) - set(C.EXTERNAL_ACCOUNTS)
            rows.append(dict(scenario_id=sid, typology=g["typology"].iloc[0], root_txn_id=g["txn_id"].iloc[0],
                             total_amount=round(float(self.meta[sid]), 2),
                             num_fragments=int((g["amount"] < C.ALERT_THRESHOLD_INR).sum()),
                             num_accounts=len(accs), start_ts=g["ts"].min(), end_ts=g["ts"].max()))
        scen = pd.DataFrame(rows)[S.SCENARIO_COLUMNS]
        return accounts, txns, scen


def generate(n_accounts: int = C.N_ACCOUNTS, seed: int = C.SEED):
    g = Generator(n_accounts, seed)
    g.build_accounts()
    g.plan_all()
    g.normal_traffic()
    g.emit_all()
    return g.finalize()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-accounts", type=int, default=C.N_ACCOUNTS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", type=Path, default=C.DATA_OUT)
    a = ap.parse_args(argv)
    acc, tx, sc = generate(a.n_accounts, a.seed)
    S.validate_accounts(acc)
    S.validate_transactions(tx, acc)
    S.validate_scenarios(sc, tx)
    a.out.mkdir(parents=True, exist_ok=True)
    acc.to_csv(a.out / "accounts.csv", index=False)
    tx.to_csv(a.out / "transactions.csv", index=False)
    sc.to_csv(a.out / "scenarios.csv", index=False)
    test = tx["ts"] >= C.SPLIT_TS
    print(f"accounts={len(acc)} transactions={len(tx)} scenarios={len(sc)}")
    print(f"label rate={tx['label'].mean():.4%} (pos={int(tx['label'].sum())})  "
          f"train={int((~test).sum())} test={int(test.sum())} test label rate={tx.loc[test, 'label'].mean():.4%}")
    print(tx[tx.label == 1].groupby("typology").size().to_string())
    return acc, tx, sc


if __name__ == "__main__":
    main()
