package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/segmentio/kafka-go"
)

type EnvironmentMessage struct {
	ServerLoads        []float64 `json:"server_loads"`
	RequestQueueLength int       `json:"request_queue_length"`
	ResponseTime       int       `json:"response_time"`
}

// Функция отправки сообщения в кафку
func sendEnvironment(){
	writer := kafka.NewWriter(kafka.WriterConfig{
		Brokers: []string{"localhost:9092"},
		Topic: "environment_topic",
	})

	envMessage := getEnvironmentState()

	jsonMessage, err:= json.Marshal(envMessage)
	if err != nil {
		log.Printf("Error encoding message to json: %v", err)
		return
	}

	err = writer.WriteMessages(context.Background(), kafka.Message{
		Key: []byte(""),
		Value: jsonMessage,
	})
	if err!=nil{
		log.Printf("Error send message to Kafka: %v", err)
		return
	}
	
	writer.Close()
}


func getEnvironmentState() EnvironmentMessage {
	return nil
}

// Функция отправки запроса на сервер
func sendRequest(url string, requestBody []byte) ([]byte, error) {
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Обработчик запросов
func handler(w http.ResponseWriter, r *http.Request) {
	// Проверяем, что метод запроса - POST
	if r.Method != http.MethodPost {
		http.Error(w, "Метод запроса должен быть POST", http.StatusMethodNotAllowed)
		return
	}

	// Получаем тело запроса
	var requestBody map[string]interface{}
	err := json.NewDecoder(r.Body).Decode(&requestBody)
	if err != nil {
		http.Error(w, fmt.Sprintf("Ошибка при декодировании тела запроса: %v", err), http.StatusBadRequest)
		return
	}

	// Пример: отправляем тело запроса на RL нейронную сеть для анализа и получения результата
	// Вместо этого используйте ваш код обращения к нейронной сети RL
	rlResponse := map[string]interface{}{
		"server": "server1", // Пример: результат от RL нейронной сети
	}

	// Определяем сервер для перенаправления запроса на основе ответа от RL нейронной сети
	serverToRedirect := rlResponse["server"].(string)

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
		http.Error(w, "Недопустимый сервер", http.StatusInternalServerError)
		return
	}

	// Преобразуем тело запроса в []byte
	requestBodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		http.Error(w, fmt.Sprintf("Ошибка при кодировании тела запроса: %v", err), http.StatusInternalServerError)
		return
	}

	// Перенаправляем запрос на выбранный сервер
	responseBody, err := sendRequest(serverURL+port, requestBodyBytes)
	if err != nil {
		http.Error(w, fmt.Sprintf("Ошибка при отправке запроса на сервер: %v", err), http.StatusInternalServerError)
		return
	}

	// Возвращаем ответ от выбранного сервера
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(responseBody)
}

func main() {	
	
	port := ":8080"

	http.HandleFunc("/", handler)

	log.Printf("Сервер запущен на порту %s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
