import json
import traceback
from pathlib import Path

import uvicorn
from description import PREDICT_DESC, api_description
from fastapi import FastAPI, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse
from server_logger import logger

import src.server.router as router
from src.server.description import MODELS_INFERENCE_TAG, RESOURCES_TAG
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

app = FastAPI(debug=True, openapi_tags=tags_metadata, description=api_description)
# app.middleware("http")(catch_exceptions_middleware)


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown")


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


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
