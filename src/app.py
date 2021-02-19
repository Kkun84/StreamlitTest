from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms


transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@st.cache
def predict(image: Image.Image) -> List[Tuple[int, str, float]]:
    with torch.no_grad():

        # https://pytorch.org/docs/stable/torchvision/models.html
        model = models.mobilenet_v2(pretrained=True)

        batch = torch.unsqueeze(transform(image), 0)

        model.eval()
        predicted = model(batch)

        with open('imagenet_classes.txt') as f:
            id, class_name = zip(*[line.strip().split(', ') for line in f.readlines()])

        probability = torch.nn.functional.softmax(predicted, dim=1)[0]
        _, indices = torch.sort(predicted, descending=True)
    return [(class_name[index], probability[index].item()) for index in indices[0]]


def make_highlighted_image(image: Image.Image, highlight_area: Dict[str, int]) -> Image:
    assert set(highlight_area) == {'top', 'bottom', 'left', 'right'}
    highlighted_image = image.copy()
    draw = ImageDraw.Draw(highlighted_image, 'RGBA')
    draw.rectangle(
        (0, 0, image.width - 1, highlight_area['top'] - 1),
        fill=(0,) * 3 + (0x80,),
    )
    draw.rectangle(
        (
            0,
            highlight_area['top'],
            highlight_area['left'] - 1,
            highlight_area['bottom'] - 1,
        ),
        fill=(0,) * 3 + (0x80,),
    )
    draw.rectangle(
        (
            highlight_area['right'],
            highlight_area['top'],
            image.width - 1,
            highlight_area['bottom'] - 1,
        ),
        fill=(0,) * 3 + (0x80,),
    )
    draw.rectangle(
        (0, highlight_area['bottom'], image.width - 1, image.height - 1),
        fill=(0,) * 3 + (0x80,),
    )
    line_width = max(min(image.width, image.height) // 50, 1)
    draw.rectangle(
        (
            highlight_area['left'] - line_width,
            highlight_area['top'] - line_width,
            highlight_area['right'] + line_width,
            highlight_area['bottom'] + line_width,
        ),
        width=line_width,
        outline=(0xFF, 0, 0, 0x80),
    )
    return highlighted_image


def main():
    st.set_page_config(
        page_title='画像分類アプリ', page_icon='🍫', initial_sidebar_state='auto'
    )

    st.title('画像分類アプリ')
    st.markdown(
        'GitHub: [https://github.com/Kkun84/StreamlitTest](https://github.com/Kkun84/StreamlitTest)'
    )

    uploaded_file = st.file_uploader('', type=['png', 'ping', 'jpg'])

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)

    input_area = {}
    with st.sidebar:
        st.title('入力する画像範囲の選択')

        input_area['left'], input_area['right'] = st.slider(
            '横の入力範囲（←左 右→）',
            min_value=0,
            max_value=image.width - 1,
            value=(0, image.width - 1),
            step=1,
        )
        input_area['top'], input_area['bottom'] = st.slider(
            '縦の入力範囲（←上 下→）',
            min_value=0,
            max_value=image.height - 1,
            value=(0, image.height - 1),
            step=1,
        )
    is_cropped = [
        input_area['top'],
        input_area['left'],
        input_area['bottom'],
        input_area['right'],
    ] == [0, 0, image.width - 1, image.height - 1]

    highlighted_image = make_highlighted_image(image, input_area)
    cropped_image = image.crop(
        (
            input_area['left'],
            input_area['top'],
            input_area['right'],
            input_area['bottom'],
        )
    ).convert('RGB')

    is_cropped = image.size == highlighted_image.size

    st.image(highlighted_image, caption='アップロード画像', use_column_width=True)
    with st.sidebar:
        st.title('画像データ')
        st.image(highlighted_image, caption='アップロード画像', use_column_width=True)
        if is_cropped:
            st.image(cropped_image, caption='入力した画像', use_column_width=True)

    predicted = predict(cropped_image)

    df = pd.DataFrame(
        predicted,
        columns=['名前', '確率[%]'],
        index=range(1, len(predicted) + 1),
    )
    df['確率[%]'] *= 100
    df['累積確率[%]'] = df['確率[%]'].cumsum()
    df = df[df['確率[%]'] >= 1]

    fig = px.bar(
        x=df['名前'],
        y=df['確率[%]'],
        labels={'x': '名前', 'y': '確率[%]'},
        range_y=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.table(df)
    return


if __name__ == "__main__":
    main()
