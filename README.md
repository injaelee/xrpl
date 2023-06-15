# xrpl
A set of tools for interacting with XRPL


```

python xrpl-cli.py -m setup-account -wn 'SOX Issuer' --domain_name www.semis.sym

-- account address: rKZf5x1AgsPMLDT4Li9jsYuugKYkL39KFm is "SOX Issuer"
-- account address: rD4qub36AnyrgkBqUymoscCSaedEKkwnqP is "SOX Distributor"
--
python xrpl-cli.py -m create-trust-line -wn 'SOX Distributor' -amt '150000|rKZf5x1AgsPMLDT4Li9jsYuugKYkL39KFm:SOX'


-- now send the new 'SOX' token to the "SOX Distributor"
--
python xrpl-cli.py -m make-payment -wn 'SOX Issuer' \
-amt '150000|rKZf5x1AgsPMLDT4Li9jsYuugKYkL39KFm:SOX' \
--destination rD4qub36AnyrgkBqUymoscCSaedEKkwnqP


-- let's create the AMM with 10000 SOX and 100 XRP
-- 
python xrpl-cli.py -m create-amm -wn 'SOX Distributor' \
--amount_pair '10000|rKZf5x1AgsPMLDT4Li9jsYuugKYkL39KFm:SOX-100000000|XRP'


-- create trust line with the SOX issuer to Granzort, the receiver
--
python xrpl-cli.py -m create-trust-line -wn 'Granzort' -amt '10000|rKZf5x1AgsPMLDT4Li9jsYuugKYkL39KFm:SOX'


-- Doraemon sends SOX to Granzort using its XRP holdings
-- account address: 'rsyixHvwgAHXYj3mzNwzu2udt18HCu51ZF' is Granzort
-- at this moment, spot exchange rate: 0.01 XRP / SOX or 10,000 XRP DROPS / SOX
-- using at most 20,000 drops
python xrpl-cli.py -m make-payment -wn 'Doraemon' \
-mx '200000|XRP' \
-amt '1|rKZf5x1AgsPMLDT4Li9jsYuugKYkL39KFm:SOX' \
--destination rsyixHvwgAHXYj3mzNwzu2udt18HCu51ZF

```