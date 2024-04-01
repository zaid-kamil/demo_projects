import gradio as gr
from ultralytics import YOLO


def yolo_inference(img_path, model_id, image_size, conf_threshold, iou_threshold):
    '''
    img_path: str, path to the image
    model_id: str, model id
    image_size: int, image size
    conf_threshold: float, confidence threshold (its the minimum confidence score for a bounding box to be considered)
    iou_threshold: float, IoU threshold (its the minimum IoU score for a bounding box to be considered) [IoU = Intersection over Union]
    '''
    return "good"


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(type="filepath", label="Image")
                model_path = gr.Dropdown(
                    label="Model",
                    choices=[
                        'yolov8n.pt',
                        'yolov8.pt',
                        'yolov8s.pt',
                    ],
                    value="yolov8n.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.4,
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                )
                yolo_infer = gr.Button(value="Inference")

            with gr.Column():
                output_numpy = gr.Image(type="numpy",label="Output")

        yolo_infer.click(
            fn=yolo_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
        )
        
        gr.Examples(
            examples=[
            ],
            fn=yolo_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
            cache_examples=True,
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Object Detection Using YOLO
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True)