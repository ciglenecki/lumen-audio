import json
import traceback
from pathlib import Path

import uvicorn
from description import PREDICT_DESC, api_description
from fastapi import Depends, FastAPI, Form, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib.style import available
from server_logger import logger

import src.config.config_defaults as cd
import src.server.controllers as controllers
import src.server.router as router
from src.server.description import MODELS_INFERENCE_TAG, RESOURCES_TAG
from src.server.middleware import dep_model_ckpt_path
from src.server.server_store import server_store


def catch_exceptions_middleware(request, call_next):
    try:
        return call_next(request)
    except Exception as err:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder(
                {
                    "message": "Internal server error. Check console log for more information."
                }
            ),
        )


tags_metadata = [
    {
        "name": RESOURCES_TAG,
        "description": "Models and datasets available for inference.",
    },
    {
        "name": MODELS_INFERENCE_TAG,
        "description": PREDICT_DESC,
    },
]

app = FastAPI(
    title="Audio prediction API",
    debug=True,
    openapi_tags=tags_metadata,
    description=api_description,
)
app.middleware("http")(catch_exceptions_middleware)
app.mount(
    "/static",
    StaticFiles(directory=Path(server_store.config.path_server, "static")),
    name="static",
)


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown")


templates = Jinja2Templates(
    directory=Path(server_store.config.path_server, "templates")
)


# @app.get("/", include_in_schema=False)
# async def docs_redirect():
#     return RedirectResponse(url="/docs")


def get_base_template_context(request: Request, files: list[UploadFile]):
    # TODO: finish this implementation and reuse this function in two other functions.
    available_models = server_store.get_available_models()
    result = []
    for file in files:
        if file.filename == "":
            continue
        # Perform any necessary processing on the file content
        # For simplicity, let's assume we extract the values directly
        data = {"a": 3, "b": 2, "c": 3}
        result.append({"filename": file.filename, "data": data})
    result = None if len(result) == 0 else result
    print(available_models)
    return {"request": request, "result": result, "available_models": available_models}


@app.get("/")
def index(request: Request):
    template_context = get_base_template_context(request, files=[])
    return templates.TemplateResponse("upload.html", context=template_context)


@app.post("/")
async def process_files(
    request: Request,
    model_ckpt_path: Path = Form(...),
    files: list[UploadFile] = [],
):
    template_context = get_base_template_context(request, files=files)
    controllers.set_server_store_model(model_ckpt_path)
    controllers.set_io_dataloader(files)
    predictions = controllers.predict_files()

    for filename, dict_inst in predictions.items():
        for instrument in dict_inst.keys():
            fullname_instrument = cd.INSTRUMENT_TO_FULLNAME[instrument]
            predictions[filename][fullname_instrument] = dict_inst.pop(instrument)

    template_context["preds"] = predictions
    template_context["all_instruments"] = cd.INSTRUMENT_TO_FULLNAME.values()
    return templates.TemplateResponse("upload.html", context=template_context)


app.include_router(router.model_router)
app.include_router(router.dataset_router)

with open(Path(Path(__file__).parent.resolve(), "openapi_spec.json"), "w+") as file:
    file.write(json.dumps(app.openapi()))
    file.close()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=server_store.args.host,
        port=server_store.args.port,
        loop="asyncio",
        reload=server_store.args.hot_reload,
    )
