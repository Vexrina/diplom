package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

type EnvironmentMessage struct {
	ServerLoads        []float64   `json:"server_loads"`
	RequestQueueLength int         `json:"request_queue_length"`
	ResponseTime       float64     `json:"response_time"`
	RequestBody        RequestBody `json:"request_body"`
}

type RewardMessage struct{
	Reward float64 `json:"reward"`
    Environment EnvironmentMessage `json:"environment"`
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

/*
Функция для отправки среды и промпта в RL модельку.
*/
func sendEnvironment(request Request) {
	envMessage := getEnvironmentState(request)

	jsonMessage, err := json.Marshal(envMessage)
	if err != nil {
		log.Printf("Error encoding message to json: %v", err)
		return
	}

	rlResponse, err := sendPostRequest("0.0.0.0:7999/predict_server", jsonMessage)
	if err != nil {
		log.Printf("Error encoding message to json: %v", err)
		return
	}

	rlResponses <- string(rlResponse)
}

/*
Функция для отправки награды и среды в RL модельку.
*/
func sendReward(reward float64, request Request) {
	envMessage := getEnvironmentState(request)

	jsonMessage, err := json.Marshal(RewardMessage{reward, envMessage})
	if err != nil {
		log.Printf("Error encoding message to json: %v", err)
		return
	}

	_, err = sendPostRequest("0.0.0.0:7999/get_reward", jsonMessage)
	if err != nil {
		log.Printf("Error encoding message to json: %v", err)
		return
	}
}

/*
# TODO: посмотреть, есть ли внутренняя функция для этого.

Функция-утилита для подсчета среднего. Используется только
для подсчета среднего по времени ответа.
*/
func average(slice []float64) float64 {
	sum := 0.0
	for _, num := range slice {
		sum += num
	}
	return sum / float64(len(slice))
}

/*
Функция утилита для посыла гетзапросов на взятие нагрузки на сервера.
Ходит на пришедший url, считывает данные, возвращает
*/
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

/*
Функция-забиратель нагрузки на серверы.
Ходит по внешним серверам(в рамках ВКР рассматриваются исключительно localhost:808(1,2,3),
но в теории можно ее масштабировать достаточно просто).
Заполняет нагрузку на сервера в слайс, высчитывает среднее время отклика и считает текущую
заполненность очереди.
Возвращает EnvironmentMessage.
*/
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

/*
Функция POST запроса по юрлу.
Отправляет запрос и ждет ответа.
Возвращает, логично, ответ.
*/
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

/*
# Обработчик запросов.

Является основным эндпоинтом проекта, т.к. через него происходит общение всего со всем,
т.е. в контексте облачных вычислений является сервером посредником между клиентом и
системой облачных комьютеров.

При появлении запроса расшифровывает его в RequestBody, после чего добавляет его в очередь запросов.
Если очередь переполнена -> http.StatusServiceUnavailable

Разбирает очередь - worker.

Как только получен ответ, возвращает его пользователю.
*/
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

/*
Используем воркера, который при появлении запроса будет запускать логику его обработки.
Он будет засекать время начала обработки, ходить в нейронку со средой и запросом,
принимать выход нейронки. После чего отсылать запрос на нужный сервер и отправлять награду
и новую среду нейронке для обновления весов.
Когда запрос будет обработан вернет его клиенту.

Запускать как горутину.
*/
func worker() {
	for req := range requestQueue {
		reward:=0.0
		start := time.Now()
		sendEnvironment(req)
		rlResponse := <-rlResponses

		// Определяем сервер для перенаправления запроса на основе ответа от RL нейронной сети
		serverToRedirect := rlResponse

		if req.Body.Video{
			if serverToRedirect=="server3"{
				reward-=1
			} else {
				reward += 0.5
			}
		} else {
			if serverToRedirect=="server1"{
				reward-=1
			} else {
				reward += 0.5
			}
		}
		// Определяем URL сервера на основе ответа от RL нейронной сети
		serverURL := "localhost:"
		port := "null"
		switch serverToRedirect {
		case "server1":
			port = "8081" // ToVideo
		case "server2":
			port = "8082" // ToBoth
		case "server3":
			port = "8083" // ToImage
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
		sendReward(reward, req)
	}
}

func main() {
	go worker()

	port := ":8080"

	http.HandleFunc("/", handler)

	log.Printf("Сервер запущен на порту %s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
