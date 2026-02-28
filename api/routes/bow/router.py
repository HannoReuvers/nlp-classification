from fastapi import APIRouter

router = APIRouter(
    prefix="/bow",
    tags=["Bag of words classifier"],
)


@router.get("")
async def root():
    return {"message": "Bag of Words API"}
