# liquotaku

Link to the [Medium post](https://quantitative-modelling-for-fun.medium.com/a-physicist-view-on-ai-ii-time-series-in-finance-e868f7462d74).

In this repo, I used go to query asynchronously the Binance API to retrieve high frequency data of traded volumes. I then used Jupyter and Keras to build Neural Networks that predicts the data well.

Check the Notebook for more info!

## How to get started

First you need to create the data folders using the command below from the root of the repo.
```
mkdir -p data/volumes data/concatenated
```

Then you need to extract the data from Binance API using, for example the command below.
```
go run go/cmd/binanceklines.go --folder data/volumes --symbol BTCUSDT --start 20231001 --end 20231101
```

My extraction script is not very user friendly so if nothing happens, it probably means that the input is ill defined. So make sure that
- the path to the data folder is valid.
- the symbol is listed on Binance.
- the start and end date are in the format YYYYMMDD.
- the start date is a date prior to the end date.

Drop me a line if you have issues. :)

ALSO! Be aware that the platform has request limits. If you don't follow them your IP could be banned temporarily (or worse!). When I queried over 6 years, I started 6 processes with a `sleep 60` command between each process launch. Here is the command I used:
```
go run go/cmd/binanceklines.go --start 20230101 --end 20231101 && sleep 60 && \
go run go/cmd/binanceklines.go --start 20220101 --end 20230101 && sleep 60 && \
go run go/cmd/binanceklines.go --start 20210101 --end 20220101 && sleep 60 && \
go run go/cmd/binanceklines.go --start 20200101 --end 20210101 && sleep 60 && \
go run go/cmd/binanceklines.go --start 20190101 --end 20200101 && sleep 60 && \
go run go/cmd/binanceklines.go --start 20180101 --end 20190101
```

At the end of the extraction run
```
ls data/volumes
```
and if it worked, you should have something similar to
```
% ls data/volumes
BTCUSDT_20231001.csv    BTCUSDT_20231006.csv    BTCUSDT_20231011.csv    BTCUSDT_20231016.csv    BTCUSDT_20231021.csv    BTCUSDT_20231026.csv    BTCUSDT_20231031.csv
BTCUSDT_20231002.csv    BTCUSDT_20231007.csv    BTCUSDT_20231012.csv    BTCUSDT_20231017.csv    BTCUSDT_20231022.csv    BTCUSDT_20231027.csv
BTCUSDT_20231003.csv    BTCUSDT_20231008.csv    BTCUSDT_20231013.csv    BTCUSDT_20231018.csv    BTCUSDT_20231023.csv    BTCUSDT_20231028.csv
BTCUSDT_20231004.csv    BTCUSDT_20231009.csv    BTCUSDT_20231014.csv    BTCUSDT_20231019.csv    BTCUSDT_20231024.csv    BTCUSDT_20231029.csv
BTCUSDT_20231005.csv    BTCUSDT_20231010.csv    BTCUSDT_20231015.csv    BTCUSDT_20231020.csv    BTCUSDT_20231025.csv    BTCUSDT_20231030.csv
```
Finally, you need to concatenate the data before running the notebook. You can use the command
```
cd data/volumes && cat $(ls *) > ../concatenated/BTCUSDT_20180101_20231101.csv && cd -
```
And check with
```
ls data/concatenated 
```
if it worked, you should have something similar to
```
% ls data/concatenated 
BTCUSDT_20180101_20231101.csv
```

Now you can play with the notebook!
