import json
from typing import Optional
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

import logging

from fastapi import Depends, HTTPException, Request, status


from src.config import settings
from fastapi import APIRouter

from src.models import DivinationBody, User
from src.user import get_user
from src.limiter import get_real_ipaddr, check_rate_limit
from src.divination import DivinationFactory

client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.api_base)
router = APIRouter()
_logger = logging.getLogger(__name__)
STOP_WORDS = [
    "忽略", "ignore", "指令", "命令", "command", "help", "帮助", "之前",
    "幫助", "現在", "開始", "开始", "start", "restart", "重新开始", "重新開始",
    "遵守", "遵循", "遵从", "遵從"
]

# 统一输出规范：让占卜结果更明确、可读、可操作
OUTPUT_GUIDELINE = (
    "\n\n【输出要求，请严格遵守】\n"
    "1. 结论先行：开篇先用一两句话直接给出明确的判断或结论，不要铺垫。\n"
    "2. 分点展开：用编号列表分点说明关键内容（如：现状、走向、需要注意的点），每点一句话讲清。\n"
    "3. 具体可操作：在解释基础上，务必给出针对提问者的、具体可执行的建议，而不是泛泛而谈。\n"
    "4. 有倾向性：基于牌面/依据给出你自己的倾向判断，避免整篇堆砌“可能”“或许”等模糊词。\n"
    "5. 语气：坦诚、直接、有温度，篇幅精炼，不重复同一意思。"
)


@router.post("/api/divination")
async def divination(
        request: Request,
        divination_body: DivinationBody,
        user: Optional[User] = Depends(get_user)
):

    real_ip = get_real_ipaddr(request)
    # rate limit when not login
    if settings.enable_rate_limit:
        if not user:
            max_reqs, time_window_seconds = settings.rate_limit
            check_rate_limit(f"{settings.project_name}:{real_ip}", time_window_seconds, max_reqs)
        else:
            max_reqs, time_window_seconds = settings.user_rate_limit
            check_rate_limit(
                f"{settings.project_name}:{user.login_type}:{user.user_name}", time_window_seconds, max_reqs
            )

    _logger.info(
        f"Request from {real_ip}, "
        f"user={user.model_dump_json(context=dict(ensure_ascii=False)) if user else None}, "
        f"body={divination_body.model_dump_json(context=dict(ensure_ascii=False))}"
    )
    if any(w in divination_body.prompt.lower() for w in STOP_WORDS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prompt contains stop words"
        )
    divination_obj = DivinationFactory.get(divination_body.prompt_type)
    if not divination_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No prompt type {divination_body.prompt_type} not supported"
        )
    prompt, system_prompt = divination_obj.build_prompt(divination_body)

    # custom api key, model and base url support
    custom_base_url = request.headers.get("x-api-url")
    custom_api_key = request.headers.get("x-api-key")
    custom_api_model = request.headers.get("x-api-model")
    api_client = client
    api_model = custom_api_model if custom_api_model else settings.model
    if custom_base_url and custom_api_key:
        api_client = AsyncOpenAI(api_key=custom_api_key, base_url=custom_base_url)
    elif custom_api_key:
        api_client = AsyncOpenAI(api_key=custom_api_key, base_url=settings.api_base)

    if not (settings.api_base or custom_base_url) or not (settings.api_key or custom_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="请设置 API KEY 和 API BASE URL"
        )

    try:
        openai_stream = await api_client.chat.completions.create(
            model=api_model,
            max_tokens=1000,
            temperature=0.9,
            top_p=1,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt + OUTPUT_GUIDELINE
                },
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:
        _logger.error(f"OpenAI API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenAI API error: {str(e)}"
        )

    async def get_openai_generator():
        try:
            async for event in openai_stream:
                if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                    current_response = event.choices[0].delta.content
                    yield f"data: {json.dumps(current_response)}\n\n"
        except Exception as e:
            _logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(get_openai_generator(), media_type='text/event-stream')
