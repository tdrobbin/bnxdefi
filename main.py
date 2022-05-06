import re
# from turtle import st
import numpy as np
import pandas as pd
import bs4
from pathlib import Path
import pandas as pd
from zmq import RATE
from playwright.sync_api import sync_playwright
from loguru import logger
from box import Box
from scipy.stats import norm


# PARENT_DIR = Path(f'C:/Users/tyler/Downloads/bnx_defi')
PARENT_DIR = Path.home() / 'bnx_defi'
PARENT_DIR.mkdir(parents=True, exist_ok=True) 

CACHE_DIR = PARENT_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BNX_STAKING_URL = 'https://www.binance.com/en/staking'
BNX_DEFI_URL = 'https://www.binance.com/en/defi-staking'

logger.add(PARENT_DIR / 'bnx_defi.log')
logger.info(f'defi directory set to: {PARENT_DIR}')

RATE_OVERRIDES_CSV = PARENT_DIR / 'rate_overrides.csv'
if not RATE_OVERRIDES_CSV.exists():
    logger.info(f'created empty rates override csv: {RATE_OVERRIDES_CSV}')
    empty_rate_overrides_df = pd.DataFrame(columns=['asset', 'apy', 'duration', 'txt', 'airdrop', 'type'])
    empty_rate_overrides_df.to_csv(RATE_OVERRIDES_CSV, index=False)


# DEFI -----------------------------------------------------------------------------

def download_staking_html(bnx_url=BNX_STAKING_URL, parent_dir=PARENT_DIR):
    now_ts = pd.Timestamp.now()
    now = now_ts.strftime('%Y%m%d_%H%M')
    html_fpath = parent_dir / (now + '_staking.html')
    csv_fpath = parent_dir / (now + '_staking.csv')
    
    with sync_playwright() as p:
        logger.info('scraping binance for staking rates')
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(bnx_url)
        page.wait_for_timeout(10000)

        page.locator('#onetrust-accept-btn-handler').click()
        page.wait_for_timeout(2000)

        # import ipdb; ipdb.set_trace()
        
        # page.locator('.css-xgkowf > div:nth-child(1) > input:nth-child(1)').click()
        # page.query_selector('.css-xgkowf > div:nth-child(1) > input:nth-child(1)').click()
        page.evaluate("document.querySelector('.css-xgkowf > div:nth-child(1) > input:nth-child(1)').click()")


        # .css-xgkowf > div:nth-child(1) > input:nth-child(1)
        page.wait_for_timeout(2000)
        page.locator('#savings-lending-pos-expend').click()
        page.wait_for_timeout(2000)
        
        for i in range(30):
            # for i in range(2):
            page.mouse.wheel(delta_x=0, delta_y=200_000)
            page.wait_for_timeout(2000)
            
            # page.wait_for_timeout(2000)
        
        logger.info(f'saving html: {html_fpath}')
        html = page.content()
        with open(html_fpath, 'w', encoding='utf-8') as f:
            f.write(html)


def download_defi_html(bnx_url=BNX_DEFI_URL, parent_dir=PARENT_DIR):
    now_ts = pd.Timestamp.now()
    now = now_ts.strftime('%Y%m%d_%H%M')
    html_fpath = parent_dir / (now + '_defi.html')
    csv_fpath = parent_dir / (now + '_defi.csv')
    
    with sync_playwright() as p:
        logger.info('scraping binance for defi rates')
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()
        page.goto(bnx_url)
        page.wait_for_timeout(10000)

        page.locator('#onetrust-accept-btn-handler').click()
        page.wait_for_timeout(2000)

        # import ipdb; ipdb.set_trace()
        
        # page.locator('.css-xgkowf > div:nth-child(1) > input:nth-child(1)').click()
        # page.query_selector('.css-xgkowf > div:nth-child(1) > input:nth-child(1)').click()
        page.evaluate("document.querySelector('.css-xgkowf > div:nth-child(1) > input:nth-child(1)').click()")


        # .css-xgkowf > div:nth-child(1) > input:nth-child(1)
        page.wait_for_timeout(2000)
        page.locator('#savings-lending-defi-expend').click()
        page.wait_for_timeout(2000)
        
        for i in range(10):
            # for i in range(2):
            page.mouse.wheel(delta_x=0, delta_y=200_000)
            page.wait_for_timeout(2000)
            
            # page.wait_for_timeout(2000)
        
        logger.info(f'saving html: {html_fpath}')
        html = page.content()
        with open(html_fpath, 'w', encoding='utf-8') as f:
            f.write(html)
    

def parse_html(fpath, site='staking'):
    logger.info('parsing html')
    with open(fpath, mode='r', encoding='cp437') as f:
        html = f.read()

    bs = bs4.BeautifulSoup(html)
    rez = list(bs.find_all('div', {'class': 'css-n1ers'}))
    ds = [parse_text(e.text, site=site) for e in rez]
    df = pd.DataFrame(ds)

    df = df[~df.airdrop]
    df = df.astype(dict(apy=float, duration=int))
    df['apy'] = df['apy'] / 100
    df = df.drop_duplicates()
    
    return df


def get_spot_bar(syms):
    logger.info('pulling spot price data from binance')

    api = SpotKlineDaily()
    df = api.fetch(secids=np.unique(syms), start=pd.Timestamp.today() - pd.offsets.Day(183), end=pd.Timestamp.today())

    now_ts = pd.Timestamp.now()
    now = now_ts.strftime('%Y%m%d_%H%M')
    
    # xl_fpath = CACHE_DIR / ('spot_' + now + '.xlsx')
    csv_fpath = CACHE_DIR / ('spot_' + now + '.csv')

    logger.info(f'caching spot data: {csv_fpath}')
    # df.to_excel(xl_fpath)
    df.to_csv(csv_fpath)

    return df
    # raise NotImplementedError


def get_perp_funding(syms):
    logger.info('pulling perp futures funding rates from binance')

    api = PerpFundingRate()
    df = api.fetch(secids=np.unique(syms), start=pd.Timestamp.today() - pd.offsets.Day(183), end=pd.Timestamp.today())

    now_ts = pd.Timestamp.now()
    now = now_ts.strftime('%Y%m%d_%H%M')
    
    xl_fpath = CACHE_DIR / ('funding_' + now + '.xlsx')
    csv_fpath = CACHE_DIR / ('funding_' + now + '.csv')

    logger.info(f'caching funding data: {csv_fpath}')
    # df.to_excel(xl_fpath)
    df.to_csv(csv_fpath)

    return df
    # raise NotImplementedError


def compute_single_asset_ret(
    yield_for_duration=.05,
    stake_ratio=.5,
    return_of_underlying=0,
    length_of_duration=90,
    funding_rate_for_duration=0
):
    """
    assume we stake stake_ratio
    then hedge stake_ratio with a short perpetual future
    so stake_ratio should be <= 0.5 and the higher the volatility of the underlying
    in fact the smaller stake_ratio you want so you can post more margin for future
    and minmize risk of liquidation.

    also assuming short s*(1+y) to minimize exposure to underlying
    """
    y = yield_for_duration
    s = stake_ratio
    r = return_of_underlying
    l = length_of_duration
    f = funding_rate_for_duration

    #      <--SHORT----->   <--LONG+YIELD------->   <--FUNDING>
    return (s)*(1+y)*(-r) + ((s)*(1+y)*(1+r) - s) - ((s)*(-f))


def get_full_yield_data(parent_dir=PARENT_DIR):
    logger.info('computing expected returns by asset')
    
    staking_fpath = sorted(parent_dir.glob('*_staking.html'))[-1]
    staking_df = parse_html(staking_fpath)
    staking_df['type'] = 'staking'

    defi_fpath = sorted(parent_dir.glob('*_defi.html'))[-1]
    defi_df = parse_html(defi_fpath, site='defi')
    defi_df['type'] = 'defi'

    df = pd.concat([staking_df, defi_df], axis=0, ignore_index=True)
    df['override'] = False
    # df = df.reset_index

    if RATE_OVERRIDES_CSV.exists():
        overrides = pd.read_csv(RATE_OVERRIDES_CSV)
        if len(overrides) >= 1:
            for idx, row in overrides.iterrows():
                df_target_loc = df[(df['asset']==row['asset']) & (df['type']==row['type'])].index[0]
                logger.info(f'appyling rate override: {row.dropna().to_dict()}')

                df.loc[df_target_loc, 'apy'] = row['apy']
                df.loc[df_target_loc, 'duration'] = row['duration']
                df.loc[df_target_loc, 'override'] = True


    df['sym'] = df.asset + 'USDT'
    df['duration_yield'] = df.apy / (365/df.duration)
    df.index = df.sym

    spot_bar = get_spot_bar(syms=df.index)
    # spot_bar = pd.read_csv(r'C:\Users\tyler\bnx_defi\cache\spot_20220407_2041.csv').set_index(['secid', 'timestamp'])
    spot_px = spot_bar.reset_index().pivot(columns='secid', values='close', index='timestamp')
    spot_px = spot_px.astype(float)

    df['daily_vol'] = spot_px.pct_change().std()
    df['duration_vol'] = df['daily_vol'] * np.sqrt(df['duration'])
    df['duration_prob_liquidation'] = df.apply(lambda x: norm(loc=0, scale=x['duration_vol']).sf(1), axis=1)

    perp_funding = get_perp_funding(syms=df.index)
    # perp_funding = pd.read_csv(r'C:\Users\tyler\bnx_defi\cache\funding_20220407_2042.csv').set_index(['secid', 'timestamp'])
    perp_funding = perp_funding.reset_index().pivot(columns='secid', values='fundingRate', index='timestamp')
    perp_funding = perp_funding.astype(float)

    def _compute_expected_duration_funding_rate(sym, duration):
        """
        funding rate is every 8 hrs, so need to multiply by 3 to get daily
        """
        if sym in perp_funding.columns:
            return perp_funding[sym].iloc[-(3 * duration):].mean() * 3 * duration
        else:
            return np.nan
    
    df['duration_fundingrate'] = np.vectorize(_compute_expected_duration_funding_rate)(df['sym'], df['duration'])

    def _compute_duration_fundingrate_r2(sym, duration):
        if sym in perp_funding.columns:
            wdw = duration * 3
            roll = perp_funding[sym].rolling(wdw).sum().dropna()
            fwd = roll.shift(-wdw).dropna()
            roll, fwd = roll.align(fwd, join='inner')
            r2 = roll.corr(fwd)**2
            return r2
        else:
            return np.nan
    
    df['duration_fundingrate_r2'] = np.vectorize(_compute_duration_fundingrate_r2)(df['sym'], df['duration'])

    df['expected_return'] = np.vectorize(compute_single_asset_ret)(
        yield_for_duration=df['duration_yield'],
        stake_ratio=.5,
        return_of_underlying=0,
        length_of_duration=df['duration'],
        funding_rate_for_duration=df['duration_fundingrate']
    )

    df['expected_return_annl'] = df['expected_return'] * (365 / df['duration'])
    df = df.sort_values('expected_return_annl', ascending=False)

    return Box(dict(
        df=df,
        price=spot_px,
        funding_rate=perp_funding,
        price_corr=spot_px.corr()
    ))


def parse_text(txt, site='staking'):
    # txt = txts[0]
    asset_overrides = '1INCH'.split(' ')
    txt = txt.replace('Stake Now' ,'')

    asset_override_applied = False
    for a in asset_overrides:
        if txt.find(a) != -1:
            asset = a
            start_str = a
            asset_override_applied = True
            break
            
    if not asset_override_applied:
        if site=='staking':
            asset_regex = r'^[A-Z]*'
            start_regex = r'^[A-Z]*'
        else:
            # import ipdb; ipdb.set_trace()
            asset_regex = r'[A-Z]*$'
            start_regex = r'^[A-Z]*'

        cregex = re.compile(asset_regex)
        rez = cregex.search(txt)
        asset = rez.group()

        cregex = re.compile(start_regex)
        rez = cregex.search(txt)
        start_str = rez.group()
        # start_asset
    
    
    apy = txt.split('%')[0].replace(start_str, '')
    # cregex = re.compile(regex)
    # rez = cregex.match(txt)

    possible_durations = '7 15 30 60 90 120'.split(' ')
    present_durations = []
    for d in possible_durations:
        if d in txt.replace(asset, '').split('%')[1]:
            present_durations.append(int(d))

    max_duration = np.max(present_durations) if len(present_durations) > 0 else 1
    
    airdrop = txt.find('AirDrop') != -1
    
    return dict(asset=asset, apy=apy, duration=max_duration, txt=txt, airdrop=airdrop)


# BNX ENDPOINTS -------------------------------------------------------------------------

import itertools
# pip install box
from box import Box


def to_posix_ms(ts, unit='ms'):
    mul_fac = {
        's': 1,
        'ms': 1e3,
        'ns': 1e6
    }

    return str(int(np.round(pd.Timestamp(ts).timestamp() * mul_fac[unit], decimals=0)))


def validate_date(date, as_str=True):
    if isinstance(date, (pd.Timestamp, datetime.date)):
        date = date.strftime('%Y%m%d')

    elif isinstance(date, int):
        date = (pd.Timestamp.now() - pd.offsets.Day(date)).strftime('%Y%m%d')

    elif isinstance(date, str):
        date = pd.to_datetime(date).strftime('%Y%m%d')

    else:
        raise ValueError(f'unable to validate date: {date}')

    if as_str:
        return date

    return pd.Timestamp(date)


def utctoday():
    return validate_date(pd.Timestamp.utcnow(), as_str=False)


def today():
    return validate_date(pd.Timestamp.now(), as_str=False)


from functools import lru_cache
import ccxt
import pandas as pd
from tqdm.auto import tqdm


@lru_cache()
def get_ccxt_binance_api(wallet='f'):
    if wallet == 'f':
        options = {'defaultType': 'future'}
    elif wallet == 'd':
        options = {'defaultType': 'delivery'}
    elif wallet == 's':
        options = {}

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': options
    })

    exchange.load_markets()

    return exchange


class SinglePeriod:
    def __init__(self, start, end):
        self.start_time = pd.Timestamp(start)
        self.end_time = pd.Timestamp(end)


class BinanceEndpoint:
    wallet = None
    func_name = None
    limit = None
    freq = None
    secid_label = 'symbol'
    columns = None
    interval = None

    options = {}

    default_start = '20150101'

    def __init__(self):
        assert pd.notnull([self.func_name, self.columns, self.wallet]).all()

    def empty_df(self):
        return self.finalize(pd.DataFrame(columns=self.columns))

    def finalize(self, df):
        return df

    def fetch(self, secids, start, end):
        api = get_ccxt_binance_api(self.wallet)

        if self.freq is not None:
            periods = pd.period_range(start, end, freq=self.freq)
        else:
            periods = [SinglePeriod(start, end)]

        if isinstance(secids, str) or len(secids) == 1:
            if isinstance(secids, str):
                _iterable = [secids]
            else:
                _iterable = secids
        else:
            _iterable = secids

        dfs = []
        tq = tqdm(_iterable, desc='securities')
        for s in tq:
            try:
                tq.set_postfix_str(s)
                tq2 = tqdm(periods, desc='dates', leave=False)
                for p in tq2:
                    tq2.set_postfix_str(p)
                    params = {
                        self.secid_label: s,
                        'startTime': to_posix_ms(p.start_time),
                        'endTime': to_posix_ms(min(p.end_time.tz_localize('utc'), pd.Timestamp.utcnow())),
                        # 'interval': interval,
                        # 'limit': self.limit
                    }
                    if self.interval is not None:
                        params.update(dict(interval=self.interval))
                    if self.limit is not None:
                        params.update(dict(limit=self.limit))
                    # if self.freq is not
                    params.update(self.options)

                    rez = getattr(api, self.func_name)(params=params)
                    dfi = pd.DataFrame(rez, columns=self.columns)
                    dfi[self.secid_label] = s

                    dfs.append(dfi)
            except Exception as err:
                logger.info(f'error for {s}: {str(err)}')
                pass

        df = pd.concat(dfs, axis=0)
        df = self.finalize(df)

        return df

    def fetch_from_latest_timestamps(self, secids, latest_timestamps=None):
        dfs = []
        for secid in tqdm(secids):
            if latest_timestamps is not None or pd.notnull(latest_timestamps):
                try:
                    secid_latest = latest_timestamps[secid]
                except KeyError:
                    secid_latest = self.default_start
            else:
                secid_latest = self.default_start

            secid_start = pd.Timestamp(secid_latest) + pd.offsets.Day()
            secid_end = utctoday() - pd.offsets.Day()

            if (utctoday() - pd.Timestamp(secid_latest)).days < 2:
                dfi = self.empty_df()
                dfs.append(dfi)
                continue

            dfi = self.fetch(secids=secid, start=secid_start, end=secid_end)
            dfs.append(dfi)

        df = pd.concat(dfs, axis=0)

        return df


class KlineBinanceEndpoint(BinanceEndpoint):
    limit = 500
    columns = 'opent open high low close volume closet quote_volume ntrades tbv tqv ignore'.split(' ')

    def finalize(self, df):
        df['timestamp'] = pd.to_datetime(df.closet, unit='ms')
        df['secid'] = df[self.secid_label]

        df = df.set_index(['secid', 'timestamp'])
        df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']]

        return df


class PerpKlineDaily(KlineBinanceEndpoint):
    wallet = 'f'
    func_name = 'fapiPublic_get_continuousklines'
    secid_label = 'pair'
    interval = '1d'
    freq = '500D'
    # limit = 500

    options = {
        'contractType': 'PERPETUAL'
    }


class SpotKlineDaily(KlineBinanceEndpoint):
    wallet = 's'
    func_name = 'public_get_klines'
    secid_label = 'symbol'
    interval = '1d'
    freq = '500D'
    # limit = 500


class PerpFundingRate(BinanceEndpoint):
    wallet = 'f'
    func_name = 'fapiPublic_get_fundingrate'
    secid_label = 'symbol'
    limit = 1000
    freq = '300D'

    columns = ['symbol', 'fundingRate', 'fundingTime']

    def finalize(self, df):
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['secid'] = df['symbol']

        df = df.set_index(['secid', 'timestamp'])['fundingRate']

        return df


class SpotKlineMinutely(KlineBinanceEndpoint):
    wallet = 's'
    func_name = 'public_get_klines'
    secid_label = 'symbol'
    interval = '1m'
    freq = '500T'


class DeliveryKlineMinutely(KlineBinanceEndpoint):
    wallet = 'd'
    func_name = 'dapiPublic_get_klines'
    secid_label = 'symbol'
    interval = '1m'
    freq = '500T'


def main(output_dir=PARENT_DIR):
    download_staking_html()
    download_defi_html()
    d = get_full_yield_data()

    now_ts = pd.Timestamp.now()
    now = now_ts.strftime('%Y%m%d_%H%M')
    fpath = output_dir / (now + '.xlsx')

    from pandas import ExcelWriter
    
    logger.info(f'writing final execel file: {fpath}')
    writer = ExcelWriter(fpath)
    for key in d:
        d[key].to_excel(writer, key)

    writer.save()
    logger.info('done')


if __name__ == '__main__':
    main()
    # download_html()
