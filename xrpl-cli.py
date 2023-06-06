from __future__ import annotations
import argparse
from collections import namedtuple
from xrpl.transaction import (
    safe_sign_and_autofill_transaction,
    send_reliable_submission,
)
from xrpl.clients import JsonRpcClient
from xrpl.models import TrustSetFlagInterface
from xrpl.models.currencies import XRP, IssuedCurrency
from xrpl.models.amounts.issued_currency_amount import IssuedCurrencyAmount
from xrpl.wallet import generate_faucet_wallet
from xrpl.wallet.main import Wallet
from xrpl.models.transactions import (
    AccountSet,
    AccountSetFlag,
    OfferCreate,
    AMMDeposit, AMMVote, AMMBid, AMMCreate,
    TrustSet,
    Payment,
)
from xrpl.models.requests import AMMInfo, AccountInfo
from xrpl.models.requests.book_offers import BookOffers
from typing import Union
import logging
import json
import yaml


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


WalletSecret = namedtuple("WalletSecret", ["seed", "sequence"])

class EnvironmentConfig:
    def __init__(self,
        name: str,
        rest_url: str,
    ):
        self.name = name
        self.rest_url = rest_url

    def get_rest_url(self) -> str:
        return self.rest_url


class CurrencyAmountParser:
    def __init__(self):
        pass

    @classmethod
    def parse_currency(cls,
        value: str,
    ) -> IssuedCurrency:
        # 2000000|{issuer_addr}:SOX
        tokens = value.split(":")
        if len(tokens) == 1:
            # this is XRP?
            if tokens[0] != "XRP":
                raise ValueError("IOU Token must accompany an issuer")
            return XRP()

        if len(tokens) == 2:
            return IssuedCurrency(
                currency = tokens[1],
                issuer = tokens[0],
            )

        raise ValueError("Currency not in the right format. Given: {}".format(value))

    @classmethod
    def parse_currency_amount(cls,
        value: str,
    ) -> Union[IssuedCurrencyAmount, str]:
        # 2000000|{issuer_addr}:SOX
        tokens = value.split("|")
        amount = tokens[0]
        currency = cls.parse_currency(tokens[1])

        if currency == XRP():
            return amount

        return IssuedCurrencyAmount(
            currency = currency.currency,
            issuer = currency.issuer,
            value = amount,
        )


class WalletCache:
    def __init__(self):
        self.wallet_dict = {}

    def add_wallet(self,
        name: str,
        wallet: Wallet,
    ):
        logger.info("[WalletCache] Adding wallet '%s'", name)
        self.wallet_dict[name] = wallet

    def get_wallet(self,
        name: str,
        raise_error_on_not_found: bool = False,
    ):
        wallet = self.wallet_dict.get(name)
        if raise_error_on_not_found and not wallet:
            raise ValueError("Wallet with name '{}' not found.".format(name))

        return wallet

    def __repr__(self) -> str:
        v = "----------------------------------\n\n"
        for name, wallet in self.wallet_dict.items():
            v += "# {} \n{}\n\n".format(name, wallet)
        v += "----------------------------------"
        return v


class WalletBuilder:
    def __init__(self):
        self.wallet_secret: WalletSecret = None
        self.xrpl_client = None

    def with_wallet_secret(self,
        wallet_secret: WalletSecret,
    ):
        self.wallet_secret = wallet_secret
        return self

    def with_xrpl_client(self,
        xrpl_client: JsonRpcClient,
    ):
        self.xrpl_client = xrpl_client
        return self

    def build(self) -> Wallet:
        if self.wallet_secret:
            return Wallet(
                seed = self.wallet_secret.seed,
                sequence = self.wallet_secret.sequence,
            )

        if not self.xrpl_client:
            raise RuntimeError("Missing XRPL Client to build a new wallet.")

        return generate_faucet_wallet(self.xrpl_client, debug = True)


class InterCurrencyAmount:
    def __init__(self,
        currency_amount: Union[IssuedCurrencyAmount, str],
        currency: str = None,
        issuer: str = None,
        value: float = None,
    ):
        if not currency_amount:
            self.inter_currency = InterCurrency(
                currency = currency,
                issuer = issuer,
            )
            self.value = value
        elif type(currency_amount) == str:
            self.inter_currency = InterCurrency(
                currency = "XRP",
                issuer = None,
            )
            self.value = int(currency_amount)
        else:
            self.inter_currency = InterCurrency(
                currency = currency_amount.currency,
                issuer = currency_amount.issuer,
            )
            self.value = float(currency_amount.value)

    def __mul__(self,
        other: InterCurrencyAmount,
    ) -> InterCurrencyAmount:
        # n x self
        return InterCurrencyAmount(
            currency_amount = None,
            currency = self.inter_currency.currency,
            issuer = self.inter_currency.issuer,
            value = self.value * other.value,
        )

    def __rmul__(self,
        other: InterCurrencyAmount,
    ) -> InterCurrencyAmount:
        # self x n
        print("__rmul__" % self.value)
        return None

    def to_xrpl_currency_amount(self) -> Union[IssuedCurrencyAmount, str]:
        if self.inter_currency.currency == "XRP":
            return str(int(self.value))

        return IssuedCurrencyAmount(
            issuer = self.inter_currency.issuer,
            currency = self.inter_currency.currency,
            value = str(self.value),
        )

    def __repr__(self) -> str:
        return f"{self.value}|{self.inter_currency}"


class InterCurrency:
    def __init__(self,
        currency: str,
        issuer: str,
    ):
        self.currency = currency
        self.issuer = issuer

    def to_xrpl_currency(self) -> IssuedCurrency:
        if self.currency == "XRP":
            return XRP()

        return IssuedCurrency(
            issuer = self.issuer,
            currency = self.currency,
        )

    def __repr__(self) -> str:
        if self.issuer == None:
            return f"{self.currency}"
        return f"{self.issuer}:{self.currency}"



def load_from_file(
    yml_path: str,
    env_name: str,
    wallet_cache: WalletCache,
):
    with open(yml_path, mode="rt", encoding="utf-8") as file:
        yml_config = yaml.safe_load(file)

    accounts = yml_config.get("accounts", [])

    for account_info in accounts:
        logger.info("building '%s'", account_info.get("name"))
        wallet_builder = WalletBuilder()
        wallet = wallet_builder.with_wallet_secret(
            WalletSecret(account_info.get("seed"), 0)
        ).build()

        wallet_cache.add_wallet(
            name = account_info.get("name"),
            wallet = wallet,
        )


    env = yml_config.get("env", {})
    env_config = env.get(env_name)

    return EnvironmentConfig(env_name, env_config.get("rpc_url"))


"""
- TickSize
  - The number of significant digits
"""
def create_account(
    xrpl_client: JsonRpcClient,
    wallet_name: str,
):
    logger.info(
        "Creating a new wallet account with name '%s'.",
        wallet_name,
    )
    wallet_builder = WalletBuilder()
    wallet = wallet_builder.with_xrpl_client(
        xrpl_client).build()
    
    logger.info(
        "[%s] seed: %s", wallet_name, wallet.seed)
    logger.info(
        "[%s] seq : %s", wallet_name, wallet.sequence)    


def setup_account(
    xrpl_client: JsonRpcClient,
    wallet: Wallet,
    domain_name: str,
    transfer_rate: int = 0,
    tick_size: int = 10,
):
    account_mutate_tx = AccountSet(
        account = wallet.classic_address,
        transfer_rate = transfer_rate,
        tick_size = tick_size,
        domain = bytes.hex(domain_name.encode("ASCII")),
        set_flag = AccountSetFlag.ASF_DEFAULT_RIPPLE, # enabled rippling by default
    )

    signed_and_prepared_account_mutate_tx = safe_sign_and_autofill_transaction(
        transaction = account_mutate_tx,
        wallet = wallet,
        client = xrpl_client,
    )
    logging.info(json.dumps(signed_and_prepared_account_mutate_tx.to_xrpl(), indent = 2))

    response = send_reliable_submission(
        signed_and_prepared_account_mutate_tx,
        xrpl_client,
    )
    logger.info(json.dumps(response.result, indent = 2))


def create_trust_line(
    xrpl_client: JsonRpcClient,
    distributor_wallet: Wallet,
    currency_amount: IssuedCurrencyAmount,
):
    """
    Creating a token involves creating a TrustSet between the 
    cold and the hot wallet accounts.

    You can say the hot wallet account is the distributor of the
    new minted tokens; the cold wallet account is the issuer.
    """

    # create the trust set between the accounts
    trust_set_tx = TrustSet(
        account = distributor_wallet.classic_address,
        limit_amount = IssuedCurrencyAmount(
            currency = currency_amount.currency,
            issuer = currency_amount.issuer,
            value = currency_amount.value,
        ),
        flags = TrustSetFlagInterface(
             tf_clear_no_ripple=True,
        ),
    )

    # the distributor account creates the trustline to the 
    # issuer
    # this means that the distributors trusts to hold the issuer's token
    # up to the specified amount
    ts_tx_prepared = safe_sign_and_autofill_transaction(
        transaction = trust_set_tx,
        wallet = distributor_wallet,
        client = xrpl_client,
    )

    """
    So what are Trustlines?
        - The balance of credit you hold from the token's issuer
        - The limit for which your account trusts the asset's issuer
    """

    logger.info(json.dumps(ts_tx_prepared.to_xrpl(), indent = 2))
    logger.info(
        "[create_token] Setting up trustline from distributor '%s' to issuer '%s: %s|%s",
        distributor_wallet.classic_address,
        currency_amount.issuer,
        currency_amount.value,
        currency_amount.currency,
    )

    response = send_reliable_submission(
        ts_tx_prepared,
        xrpl_client,
    )
    logger.info(json.dumps(response.result, indent = 2))


def make_payment(
    xrpl_client: JsonRpcClient,
    source_wallet: Wallet,
    target_address: str,
    currency_amount: Union[IssuedCurrencyAmount, str],
    send_max: Union[IssuedCurrencyAmount, str],
):
    send_token_tx = Payment(
        account = source_wallet.classic_address,
        destination = target_address,
        amount = currency_amount,
        send_max = send_max,
    )
    pay_prepared = safe_sign_and_autofill_transaction(
        transaction = send_token_tx,
        wallet = source_wallet,
        client = xrpl_client,
    )
    logging.debug(json.dumps(pay_prepared.to_xrpl(), indent = 2))


    logger.info(
        "[send_asset] Send '%s', at most '%s', of token '%s:%s' to target '%s' from source '%s'",
        currency_amount.value if type(currency_amount) != str else currency_amount,
        send_max if send_max else "N/A",
        currency_amount.issuer if type(currency_amount) != str else "",
        currency_amount.currency if type(currency_amount) != str else "XRP",
        target_address,
        source_wallet.classic_address
    )
    response = send_reliable_submission(
        pay_prepared,
        xrpl_client,
    )
    logger.info(json.dumps(response.result, indent = 2))


def create_offer(
    xrpl_client: JsonRpcClient,
    source_wallet: Wallet,
    order_type: str, # sell | buy
    price_amount: Union[IssuedCurrencyAmount, str],
    quantity_amount: Union[IssuedCurrencyAmount, str],
):
    """
    When we're selling, the selling token in "quantity_amount" is the
    one that the taker gets.
      - taker_gets: quantity_amount
      - taker_pays: price_amount.value x quantity_amount.value (in terms of price_amount.currency)

    When we're buying, the buying token in "quantity_amount" is the one
    that the taker pays.
      - taker_gets: price_amount.value x quantity_amount.value (in terms of price_amount.currency)
      - taker_pays: quantity_amount

    str x amt
    """
    inter_price_amount = InterCurrencyAmount(price_amount)
    inter_quantity_amount = InterCurrencyAmount(quantity_amount)

    if order_type == "sell":
        inter_taker_gets_amount = inter_quantity_amount
        inter_taker_pays_amount = inter_price_amount * inter_quantity_amount
    elif order_type == "buy":
        inter_taker_gets_amount = inter_price_amount * inter_quantity_amount
        inter_taker_pays_amount = inter_quantity_amount
    else:
        raise ValueError(f"order type '{order_type}' is not recognized")    

    offer_create_tx = OfferCreate(
        account = source_wallet.classic_address,
        taker_gets = inter_taker_gets_amount.to_xrpl_currency_amount(),
        taker_pays = inter_taker_pays_amount.to_xrpl_currency_amount(),
    )

    offer_prepared = safe_sign_and_autofill_transaction(
        transaction = offer_create_tx,
        wallet = source_wallet,
        client = xrpl_client,
    )
    logger.info(json.dumps(offer_prepared.to_xrpl(), indent = 2))

    response = send_reliable_submission(
        offer_prepared,
        xrpl_client,
    )
    logger.info(json.dumps(response.result, indent = 2))


def lookup_orderbook(
    xrpl_client: JsonRpcClient,
    price_currency: IssuedCurrency,
    quantity_currency: IssuedCurrency,
):
    # book offers of the asks
    book_offers_req = BookOffers(
        taker_gets = quantity_currency,
        taker_pays = price_currency,
    )
    resp = xrpl_client.request(book_offers_req)

    if not resp.is_successful():
        print(f"[order_book] error calling asks side of the order book for '{arg}' pair.")
        return

    def __sep(order):
        if type(order) == dict:
            return order.get("currency"), float(order.get("value"))
        return "XRP", int(order) / 1_000_000.

    print("# Asks / Offers")
    for offer in reversed(resp.result.get("offers", [])):
        gets_token, gets_value = __sep(offer.get("TakerGets"))
        pays_token, pays_value = __sep(offer.get("TakerPays"))
        quality = offer.get("quality")
        price = pays_value / gets_value
        print(f"{gets_value} {gets_token} @ {price} {pays_token}")

    # book offers of the bids
    book_offers_req = BookOffers(
        taker_gets = price_currency,
        taker_pays = quantity_currency,
    )
    resp = xrpl_client.request(book_offers_req)

    if not resp.is_successful():
        print(f"[order_book] error calling bids side of the order book for '{arg}' pair.")
        return

    print("\n# Bids")
    for offer in resp.result.get("offers"):
        gets_token, gets_value = __sep(offer.get("TakerGets"))
        pays_token, pays_value = __sep(offer.get("TakerPays"))
        quality = offer.get("quality")
        price =  gets_value / pays_value
        print(f"{pays_value} {pays_token} @ {price} {gets_token}")


def create_amm(
    xrpl_client: JsonRpcClient,
    creator_wallet: Wallet,
    asset_amount_1: IssuedCurrencyAmount,
    asset_amount_2: IssuedCurrencyAmount,
    trading_fee: int,
):
    amm_create_tx = AMMCreate(
        account = creator_wallet.classic_address,
        amount = asset_amount_1,
        amount2 = asset_amount_2,
        trading_fee = trading_fee,
    )

    signed_and_prepared_amm_create_tx = safe_sign_and_autofill_transaction(
        transaction = amm_create_tx,
        wallet = creator_wallet,
        client = xrpl_client,
    )
    logging.debug(json.dumps(signed_and_prepared_amm_create_tx.to_xrpl(), indent = 2))

    response = send_reliable_submission(
        signed_and_prepared_amm_create_tx,
        xrpl_client,
    )
    logger.info(json.dumps(response.result, indent = 2))


class AMMDepositCommand:
    """
    Options to deposit into the AMM:
      - LP TOKEN
      - SINGLE ASSET
      - TWO ASSET

      - ONE ASSET + LP TOKEN: 
        Deposit up to the specified amount of one asset, 
        so that you receive exactly the specified amount 
        of LP Tokens in return (after fees).

      - LIMIT LP TOKEN
        Deposit up to the specified amount of one asset,
        but pay no more than the specified effective price
        per LP Token (after fees).
    """
    def execute_with_asset(self,
        user_wallet: Wallet,
        asset_amount: IssuedCurrencyAmount,
        asset_2: IssuedCurrency,
    ):
        tx = AMMDeposit(
            account = user_wallet.classic_address,
            asset = _ASSET,
            asset2 = _ASSET2,
            lp_token_out=IssuedCurrencyAmount(
                currency=_LPTOKEN_CURRENCY,
                issuer=_LPTOKEN_ISSUER,
                value=_AMOUNT,
            ),
            flags=AMMDepositFlag.TF_LP_TOKEN,
        )

    def execute_with_assets(self,
        user_wallet: Wallet,
        asset_amount_1: IssuedCurrencyAmount,
        asset_amount_2: IssuedCurrencyAmount,
    ):
        pass

    def execute_with_desired_lptokens(self,
        user_wallet: Wallet,
        asset_1: IssuedCurrency,
        asset_2: IssuedCurrency,
        lp_token_bid_min: IssuedCurrencyAmount,
        lp_token_bid_max: IssuedCurrencyAmount,
        auth_account_addresses: List[str],
    ):
        pass 

    def execute_with_max_asset_and_desired_lp_tokens(self,
        user_wallet: Wallet,
        asset_amount_1: IssuedCurrencyAmount,
        asset_2: IssuedCurrency,
    ):
        pass

    def execute_with_max_asset_within_effective_price(self,
    ):
        pass


class AMMBidCommand:

    def execute(self,
        user_wallet: Wallet,
        asset_1: IssuedCurrency,
        asset_2: IssuedCurrency,
        lp_token_bid_min: IssuedCurrencyAmount,
        lp_token_bid_max: IssuedCurrencyAmount,
        auth_account_addresses: List[str],
    ):
        """
        TODO: Live testing with the AMM BID
        """
        auth_accounts = []
        for addr in auth_account_addresses:
            auth_accounts.append(AuthAccount(addr))

        amm_bid_tx = AMMBid(
            account = user_wallet.classic_address,
            asset = asset_1,
            asset2 = asset_2,
            bid_min = lp_token_bid_max,
            bid_max = lp_token_bid_min,
            auth_accounts = auth_accounts
        )

        signed_and_prepared_amm_bid_tx = safe_sign_and_autofill_transaction(
            transaction = amm_bid_tx,
            wallet = creator_wallet,
            client = xrpl_client,
        )
        logging.debug(json.dumps(signed_and_prepared_amm_bid_tx.to_xrpl(), indent = 2))

        response = send_reliable_submission(
            signed_and_prepared_amm_bid_tx,
            xrpl_client,
        )
        logger.info(json.dumps(response.result, indent = 2))


def withdraw_from_amm():
    pass

def vote_for_rate_in_amm():
    pass


def lookup_amm(
    xrpl_client: JsonRpcClient,
    asset_currency_1: Union[IssuedCurrency, XRP],
    asset_currency_2: Union[IssuedCurrency, XRP],
):
    amm_info_req = AMMInfo(
        asset = asset_currency_1,
        asset2 = asset_currency_2,
    )
    logger.info(json.dumps(amm_info_req.to_dict(), indent = 2))
    
    response = xrpl_client.request(amm_info_req)
    logger.info(json.dumps(response.result, indent = 2))


def lookup_account(
    xrpl_client: JsonRpcClient,
    account_address: str,
):
    account_info_req = AccountInfo(
        account = account_address,
    )
    logger.info(json.dumps(account_info_req.to_dict(), indent = 2))
    
    response = xrpl_client.request(account_info_req)
    logger.info(json.dumps(response.result, indent = 2))


def parse_arguments() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "-m",
        "--mode",
        help = "execution mode",
        choices = [
            "create-account",
            "load-accounts",
            "create-amm",
            "create-trust-line",
            "make-payment",
            "create-offer",
            "lookup-orderbook",
            "lookup-amm",
            "lookup-account",
            "setup-account",
        ],
        type = str,
        default = "load-accounts"
    )

    arg_parser.add_argument(
        "-e",
        "--env",
        help = "environment to run against",
        type = str,
        choices = [
            "ammdevnet",
            "amm",
            "mainnet"
        ],
        default = "ammdevnet"
    )

    arg_parser.add_argument(
        "-c",
        "--config",
        help = "specificy the YML config path",
        type = str,
        default = "xrpl.yml"
    )

    arg_parser.add_argument(
        "-tm",
        "--target_amount",
        help = "specify the amount",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-sm",
        "--source_amount",
        help = "specify the amount",
        type = str,
        default = "",   
    )

    arg_parser.add_argument(
        "-r",
        "--receiver",
        help = "specify the receiver",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-w",
        "--wallet",
        help = "wallet / account address",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-wn",
        "--wallet_name",
        help = "wallet / account alias name in the config",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-p",
        "--currency_pair",
        help = "the currency pair {issuer}:{token}-{issuer}:{token}",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-amt",
        "--amount",
        help = "the interested currency amount ie. {amt}|{issuer}:{token}",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-mx",
        "--max_amount",
        help = "the max amount, usually used with the payment mode",
        type = str,
        default = "",   
    )

    arg_parser.add_argument(
        "-s",
        "--save",
        action = "store_true",
        help = "flag to indicate whether to save the resulting output of the exeuction",
    )

    arg_parser.add_argument(
        "-dest",
        "--destination",
        help = "the destination wallet account address",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-otype",
        "--order_type",
        choices = [
            "buy",
            "sell",
        ],
        help = "the order type for offer",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-amtp",
        "--amount_pair",
        help = "the amount pair {amt}|{issuer}:{token}-{amt}|{issuer}:{token}",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-qty",
        "--quantity",
        help = "the quantity amount",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-pri",
        "--price",
        help = "the price amount",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-sym",
        "--symbol",
        help = "the currency pair",
        type = str,
        default = "",
    )

    arg_parser.add_argument(
        "-dname",
        "--domain_name",
        help = "the domain name for the account",
        type = str,
        default = "",
    )
    

    return arg_parser.parse_args()


"""

-m create-account -wn "" --save
-m status-amm -p XRP:SOX
-m create-amm -sm 1000|XRP -tm 200|SOX
-wn doraemon -m payment -sm 2000|XRP -tm 1000|SOX -r rPtQXgXiJCqXyFqRcUZwtXdme2AfiTHiEF -mx 1000|XRP
-wn "Token Distributor" -m create-token -amt 2000000|{issuer_addr}:SOX
-m setup-account
-m create-offer -sm 1000|XRP -tm 300|SOX

-m make-payment -wn "SOX Issuer" -amt "150000|rPtQXgXiJCqXyFqRcUZwtXdme2AfiTHiEF:SOX" \
--destination rh6BLUz2BcqaiaPvGQ5uePBSqFW3YL8X48

-m setup-trustline 

# sell 10 XRP's with each XRP priced at 100 SOX
-m create-order -wn Doraemon --order_type sell --quantity "10|XRP" --price "100|rPtQXgXiJCqXyFqRcUZwtXdme2AfiTHiEF:SOX"
-m create-offer -wn Doraemon --order_type buy --price "1|XRP" --quantity "100|rPtQXgXiJCqXyFqRcUZwtXdme2AfiTHiEF:SOX"

-m lookup-orderbook --symbol "rPtQXgXiJCqXyFqRcUZwtXdme2AfiTHiEF:SOX|XRP"
-m create-amm --amount_pair "10|rPtQXgXiJCqXyFqRcUZwtXdme2AfiTHiEF:SOX-10|XRP"

"""

if __name__ == "__main__":
    args = parse_arguments()
    wallet_cache = WalletCache()
    
    env_config = load_from_file(
        args.config,
        args.env,
        wallet_cache,
    )
    xrpl_client = JsonRpcClient(env_config.get_rest_url())

    if args.mode == "load-accounts":        
        print(wallet_cache)

    elif args.mode == "create-account":
        create_account(xrpl_client, args.wallet_name)

    elif args.mode == "create-amm":
        tokens = args.amount_pair.split("-")
        asset1_amount = CurrencyAmountParser.parse_currency_amount(tokens[0])
        asset2_amount = CurrencyAmountParser.parse_currency_amount(tokens[1])
        create_amm(
            xrpl_client = xrpl_client,
            creator_wallet = wallet_cache.get_wallet(args.wallet_name, True),
            asset_amount_1 = asset1_amount,
            asset_amount_2 = asset2_amount,
            trading_fee = 0,
        )

    elif args.mode == "create-trust-line":
        parsed_currency_amount = CurrencyAmountParser.parse_currency_amount(args.amount)
        create_trust_line(
            xrpl_client = xrpl_client,
            distributor_wallet = wallet_cache.get_wallet(args.wallet_name, True),
            currency_amount = parsed_currency_amount,
        )

    elif args.mode == "setup-account":
        # -m setup-account -wn NAME --domain_name www.semis.sym
        setup_account(
            xrpl_client,
            wallet = wallet_cache.get_wallet(args.wallet_name, True),
            domain_name = args.domain_name,
        )

    elif args.mode == "make-payment":
        parsed_currency_amount = CurrencyAmountParser.parse_currency_amount(args.amount)
        send_max_amount = CurrencyAmountParser.parse_currency_amount(args.max_amount) if args.max_amount else None
        make_payment(
            xrpl_client = xrpl_client,
            source_wallet = wallet_cache.get_wallet(args.wallet_name, True),
            target_address = args.destination,
            currency_amount = parsed_currency_amount,
            send_max = send_max_amount,
        )

    elif args.mode == "create-offer":
        parsed_price_amount = CurrencyAmountParser.parse_currency_amount(args.price)
        parsed_quantity_amount = CurrencyAmountParser.parse_currency_amount(args.quantity)
        create_offer(
            xrpl_client = xrpl_client,
            source_wallet = wallet_cache.get_wallet(args.wallet_name, True),
            order_type = args.order_type,
            price_amount = parsed_price_amount,
            quantity_amount = parsed_quantity_amount,
        )

    elif args.mode == "lookup-orderbook":
        tokens = args.symbol.split("|")
        parsed_price_currency = CurrencyAmountParser.parse_currency(tokens[0])
        parsed_quantity_currency = CurrencyAmountParser.parse_currency(tokens[1])
        lookup_orderbook(
            xrpl_client = xrpl_client,
            price_currency = parsed_price_currency,
            quantity_currency = parsed_quantity_currency,
        )

    elif args.mode == "lookup-amm":
        tokens = args.currency_pair.split("-")
        asset1_currency = CurrencyAmountParser.parse_currency(tokens[0])
        asset2_currency = CurrencyAmountParser.parse_currency(tokens[1])
        lookup_amm(
            xrpl_client = xrpl_client,
            asset_currency_1 = asset1_currency,
            asset_currency_2 = asset2_currency,
        )

    elif args.mode == "lookup-account":
        lookup_account(
            xrpl_client = xrpl_client,
            account_address = wallet_cache.get_wallet(args.wallet_name, True).classic_address,
        )