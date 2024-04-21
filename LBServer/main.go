package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/segmentio/kafka-go"
)

type EnvironmentMessage struct {
	ServerLoads        []float64   `json:"server_loads"`
	RequestQueueLength int         `json:"request_queue_length"`
	ResponseTime       float64     `json:"response_time"`
	RequestBody        RequestBody `json:"request_body"`
}

type RequestBody struct {
	Prompt string `json:"prompt"`
	Video  bool   `json:"video"`
}

type Request struct {
	Body RequestBody
	Resp chan Response
}

type Response struct {
	Body []byte
	Err  error
}

var maxLenQueue = 50
var requestQueue = make(chan Request, maxLenQueue)

var responseTime []float64

var rlResponses = make(chan string)

// Функция отправки сообщения в кафку
func sendEnvironment(request Request) {
	writer := kafka.NewWriter(kafka.WriterConfig{
		Brokers: []string{"localhost:9092"},
		Topic:   "environment_topic",
	})

	envMessage := getEnvironmentState(request)

	jsonMessage, err := json.Marshal(envMessage)
	if err != nil {
		log.Printf("Error encoding message to json: %v", err)
		return
	}

	err = writer.WriteMessages(context.Background(), kafka.Message{
		Key:   []byte(""),
		Value: jsonMessage,
	})
	if err != nil {
		log.Printf("Error send message to Kafka: %v", err)
		return
	}

	writer.Close()
}

func readRLResponse() {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:   []string{"localhost:9092"},
		Topic:     "rl_response",
		Partition: 0,
		MinBytes:  10e3, // 10KB
		MaxBytes:  10e6, // 10MB
	})

	defer reader.Close()

	for {
		msg, err := reader.ReadMessage(context.Background())
		if err != nil {
			log.Printf("Error reading message from Kafka: %v", err)
			continue
		}

		message := string(msg.Value)
		rlResponses <- message
	}
}

func average(slice []float64) float64 {
	sum := 0.0
	for _, num := range slice {
		sum += num
	}
	return sum / float64(len(slice))
}

func sendGetRequest(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}

func getEnvironmentState(request Request) EnvironmentMessage {
	serverURL := "localhost:"
	ports := []string{"8081", "8082", "8083"}
	var serverLoads []float64
	for _, port := range ports {
		out, err := sendGetRequest(serverURL + port + "/load")
		if err != nil {
			fmt.Printf("Ошибка при отправке запроса на сервер: %v\n", err)
			continue
		}

		var load float64
		_, err = fmt.Sscanf(string(out), "Server: %s, GPUload: %f", &port, &load)
		if err != nil {
			fmt.Printf("Ошибка при парсинге ответа от сервера: %v\n", err)
			continue
		}

		serverLoads = append(serverLoads, load)
	}
	meanResponseTime := average(responseTime)
	requestQueueLength := len(requestQueue) / 50

	return EnvironmentMessage{serverLoads, requestQueueLength, meanResponseTime, request.Body}
}

// Функция отправки запроса на сервер
func sendPostRequest(url string, requestBody []byte) ([]byte, error) {
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Обработчик запросов
func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Метод запроса должен быть POST", http.StatusMethodNotAllowed)
		return
	}

	// Получаем тело запроса
	var requestBody RequestBody
	err := json.NewDecoder(r.Body).Decode(&requestBody)
	if err != nil {
		http.Error(w, fmt.Sprintf("Ошибка при декодировании тела запроса: %v", err), http.StatusBadRequest)
		return
	}

	select {
	case requestQueue <- Request{Body: requestBody, Resp: make(chan Response)}:
		// Запрос добавлен в очередь
	default:
		http.Error(w, "Очередь запросов переполнена", http.StatusServiceUnavailable)
		return
	}

	req := <-requestQueue
	defer close(req.Resp)

	// Отправляем запрос в очередь
	requestQueue <- req

	// Получаем ответ из очереди
	resp := <-req.Resp

	if resp.Err != nil {
		http.Error(w, fmt.Sprintf("Ошибка при отправке запроса на сервер: %v", resp.Err), http.StatusInternalServerError)
		return
	}

	// Возвращаем ответ от выбранного сервера
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(resp.Body)
}

func worker() {
	for req := range requestQueue {
		start := time.Now()
		sendEnvironment(req)
		rlResponse := <-rlResponses

		// Определяем сервер для перенаправления запроса на основе ответа от RL нейронной сети
		serverToRedirect := rlResponse

		// Определяем URL сервера на основе ответа от RL нейронной сети
		serverURL := "localhost:"
		port := "null"
		switch serverToRedirect {
		case "server1":
			port = "8081"
		case "server2":
			port = "8082"
		case "server3":
			port = "8083"
		default:
			req.Resp <- Response{Err: fmt.Errorf("недопустимый сервер")}
			continue
		}

		// Преобразуем тело запроса в []byte
		requestBodyBytes, err := json.Marshal(req.Body)
		if err != nil {
			req.Resp <- Response{Err: fmt.Errorf("ошибка при кодировании тела запроса: %v", err)}
			continue
		}

		// Перенаправляем запрос на выбранный сервер
		responseBody, err := sendPostRequest(serverURL+port, requestBodyBytes)
		if err != nil {
			req.Resp <- Response{Err: fmt.Errorf("ошибка при отправке запроса на сервер: %v", err)}
			continue
		}

		// Отправляем ответ в канал
		req.Resp <- Response{Body: responseBody}
		responseTime = append(responseTime, float64(time.Since(start)))
	}
}

func main() {
	go worker()
	go readRLResponse()

	port := ":8080"

	http.HandleFunc("/", handler)

	log.Printf("Сервер запущен на порту %s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
