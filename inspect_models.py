"""
Script để khám phá thông tin của các model Keras
Chạy script này trước khi build API để hiểu input/output shape của từng model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_FILES = [
    "best_model_vgg_finetune.keras",
    "best_model_vgg_transfer.keras",
    "best_model_1.keras",
    "best_model_2.keras",
]

def inspect_model(model_path):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        info = {
            "name": os.path.basename(model_path),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "num_layers": len(model.layers),
            "total_params": model.count_params(),
            "last_layer_activation": str(model.layers[-1].get_config().get("activation", "unknown")),
        }
        return info
    except Exception as e:
        return {"name": os.path.basename(model_path), "error": str(e)}

if __name__ == "__main__":
    results = []
    for fname in MODEL_FILES:
        path = os.path.join(MODEL_DIR, fname)
        print(f"Inspecting {fname}...")
        info = inspect_model(path)
        results.append(info)
        print(json.dumps(info, indent=2))
        print("-" * 60)
    
    print("\n=== SUMMARY ===")
    for r in results:
        if "error" not in r:
            print(f"{r['name']}: input={r['input_shape']} | output={r['output_shape']} | activation={r['last_layer_activation']}")
        else:
            print(f"{r['name']}: ERROR - {r['error']}")
