api_description = """
# Before consuming the endpoints set the variable `MODEL_DIRECTORY` in `.env` to a directory that contains model checkpoints (.ckpt). Models outside of this directory can't be used for inference via endpoints.

Command for running the server:
### `(venv) username@pc:~/lumen-geoguesser$ python3 src/app/main.py`

"""

predict_desc = """Endpoints used for predicting latitude and longitudes for given data.
Curl example for multiple images:

```
curl -i \\
-F "images=@data/raw/images/train/e788b3d1-9d20-466c-9dee-97982f0f9a3b/0.jpg" \\
-F "images=@data/raw/images/train/e788b3d1-9d20-466c-9dee-97982f0f9a3b/0.jpg" \\
http://0.0.0.0:8090/model/Golf_76__haversine_0.0098__val_acc_0.47__val_loss_1.98__05-04-03-36-32/predict-images
```
"""

get_models_desc = """Returns names of all available models on the server. You must use model names for all `POST` request predictions. Model name is a stem of the model checkpoint, e.g. model with filename `my_model.ckpt` has the name `my_model`. Models are fetched from the directory `MODEL_DIRECTORY` which is defined in the `.env` file. Only models with the extension `MODEL_EXTENSION` are fetched. Model names will be returned instead of the model filenames.
"""

predict_images_desc = """Infers latitude and longitude for multiple images. If you have a group of images where each image represents one cardinal direction (north, east, south, and west) ("0.jpg", ... , "270.jpg") you should use the `/model/{model_name}/predict-cardinal-images endpoint
"""
predict_cardinal_desc = """Infers latitude and longitude for a single location which is defined by exactly 4 images, each for one cardinal direction. Exactly 4 images must be sent and each image filename must match image's cardinal direction ("0.jpg", "90.jpg", "180.jpg", "270.jpg"). E.g northen image should be named "0.jpg". This structure is the same as the structure of the original dataset.
"""

predict_directory_desc = """Infers latitude and longitude for all images in the directory (`dataset_directory_path`) which contains subdirectories (uuid) with images for each cardinal direction. This structure is the same as the structure of the original dataset. Exactly 4 images must be sent per subdirectory and each image filename must match image's cardinal direction ("0.jpg", "90.jpg", "180.jpg", "270.jpg"). E.g northen image should be named "0.jpg". If `csv_filename` is provided, results will also be saved to a .csv file.
"""
