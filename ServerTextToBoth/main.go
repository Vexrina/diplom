package main

// #cgo CFLAGS: -I../
// #cgo LDFLAGS: -L../ -lpython3.8
// #include "../python_wrapper.h"
import "C"

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func callTtoI(prompt string) string {
	C.initializePython()
	defer C.finalizePython()
	pythonCode := fmt.Sprintf(`
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipe.to(torch_device="cuda:1", torch_dtype=torch.float32)
prompt = "%s"
num_inference_steps = 4 
images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=50, output_type="pil").images
print("Success")
`, prompt)
	return C.GoString(C.callPythonCode(C.CString(pythonCode)))
}

func callTtoV(prompt string) string {
	C.initializePython()
	defer C.finalizePython()
	pythonCode := fmt.Sprintf(`
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.to(torch_device="cuda:1", torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
prompt = "%s"
video_frames = pipe(prompt=prompt, num_inference_steps=25).frames[0]
print("Success")
`, prompt)
	return C.GoString(C.callPythonCode(C.CString(pythonCode)))
}

func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Метод запроса должен быть POST", http.StatusMethodNotAllowed)
		return
	}

	if err := r.ParseForm(); err != nil {
		http.Error(w, fmt.Sprintf("Ошибка при парсинге запроса: %v", err), http.StatusBadRequest)
		return
	}
	prompt := r.FormValue("prompt")
	videoStr := r.FormValue("video")

	video, err := strconv.ParseBool(videoStr)
	if err != nil {
		http.Error(w, fmt.Sprintf("Ошибка при преобразовании параметра 'video': %v", err), http.StatusBadRequest)
		return
	}
	output := ""
	if video {
		output = callTtoV(prompt)
	} else {
		output = callTtoI(prompt)
	}

	jsonMessage, err := json.Marshal(output)
	if err != nil {
		log.Printf("Error encoding message to json: %v", err)
		return
	}

	w.Header().Set("Content-Type", "text/plain")
	w.WriteHeader(http.StatusOK)
	w.Write(jsonMessage)
}

func loadHandler(w http.ResponseWriter, _ *http.Request) {
	pid := os.Getpid()

	out, err := exec.Command("nvidia-smi", "--query-compute-apps=pid,utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to execute nvidia-smi: %v", err), http.StatusInternalServerError)
		return
	}

	lines := strings.Split(string(out), "\n")
	for _, line := range lines {
		fields := strings.Split(line, ",")
		if len(fields) >= 2 && fields[0] == strconv.Itoa(pid) {
			gpuUtilization := strings.TrimSpace(fields[1])

			gpuUtilizationPercent, err := strconv.ParseFloat(gpuUtilization, 64)
			if err != nil {
				http.Error(w, fmt.Sprintf("Failed to parse GPU utilization: %v", err), http.StatusInternalServerError)
				return
			}

			gpuUtilizationDecimal := gpuUtilizationPercent / 100.0

			fmt.Fprintf(w, "Server: 8082, GPUload: %.4f\n", gpuUtilizationDecimal)
			return
		}
	}
	fmt.Fprintf(w, "Server: 8082, GPUload: 0.00\n")
}

func main() {
	port := ":8082"

	http.HandleFunc("/", handler)
	http.HandleFunc("/load", loadHandler)

	log.Printf("Сервер запущен на порту %s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
