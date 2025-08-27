from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="API_KEY"
)

result = CLIENT.infer(your_image.jpg, model_id="football-pitch-segmentation/1")
