package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

const binanceAPI = "https://api.binance.com/api/v3/klines"

func getBinanceData(symbol string, interval string, startTime, endTime int64) ([][2]float64, error) {
	url := fmt.Sprintf("%s?symbol=%s&interval=%s&startTime=%d&endTime=%d&limit=1000", binanceAPI, symbol, interval, startTime, endTime)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data [][12]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}

	result := make([][2]float64, 0, len(data))

	for _, kline := range data {
		openTime := int64(kline[0].(float64))
		volumeStr, ok := kline[5].(string)
		if !ok {
			return nil, fmt.Errorf("unexpected data format for volume")
		}
		volumeFloat, err := strconv.ParseFloat(volumeStr, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse volume: %v", err)
		}
		result = append(result, [2]float64{float64(openTime), volumeFloat})

	}

	return result, nil
}

func worker(wg *sync.WaitGroup, folder, symbol, date string) {
	defer wg.Done()

	startTime, _ := time.Parse("20060102", date)
	endTime := startTime.Add(12 * time.Hour) // extending to 24 hours

	fileName := fmt.Sprintf("%s/%s_%s.csv", folder, symbol, date)
	file, err := os.Create(fileName)
	if err != nil {
		log.Printf("Error creating file %s: %v", fileName, err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Splitting the 24 hours into two 12-hour intervals
	for i := 0; i < 2; i++ {
		data, err := getBinanceData(symbol, "30m", startTime.Unix()*1000, endTime.Unix()*1000)
		if err != nil {
			log.Printf("Error fetching data for %s: %v", date, err)
			return
		}

		for i, record := range data {
			// Skip writing the last record
			if i == len(data)-1 {
				continue
			}
			writer.Write([]string{strconv.FormatFloat(record[0], 'f', 0, 64), strconv.FormatFloat(record[1], 'f', 8, 64)})
		}

		// Move to the next 12-hour interval
		startTime = startTime.Add(12 * time.Hour)
		endTime = endTime.Add(12 * time.Hour)
	}

}

func main() {
	folder := flag.String("folder", "data/volumes", "Output folder for CSV files")
	symbol := flag.String("symbol", "BTCUSDT", "Ticker symbol")
	startDate := flag.String("start", "20200101", "Start date in format YYYYMMDD")
	endDate := flag.String("end", "20200101", "End date in format YYYYMMDD")
	flag.Parse()

	start, err := time.Parse("20060102", *startDate)
	if err != nil {
		log.Fatalf("Invalid start date: %v", err)
	}
	end, err := time.Parse("20060102", *endDate)
	if err != nil {
		log.Fatalf("Invalid end date: %v", err)
	}

	var wg sync.WaitGroup
	for ; start.Before(end); start = start.Add(24 * time.Hour) {
		wg.Add(1)
		go worker(&wg, *folder, *symbol, start.Format("20060102"))
	}
	wg.Wait()
}
