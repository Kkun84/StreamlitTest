import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
from torchvision import models, transforms
import torch
import plotly.express as px


@st.cache
def predict(pillow_image):
    with torch.no_grad():
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # https://pytorch.org/docs/stable/torchvision/models.htmlv2(pretrained=True)
        model = models.mobilenet_

        batch = torch.unsqueeze(transform(pillow_image), 0)

        model.eval()
        predicted = model(batch)

        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        prob = torch.nn.functional.softmax(predicted, dim=1)[0]
        _, indices = torch.sort(predicted, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0]]


def main():
    st.title('画像分類アプリ')

    uploaded_file = st.file_uploader('', type=['png', 'ping', 'jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        width_range = st.sidebar.slider(
            '横',
            min_value=0,
            max_value=image.width - 1,
            value=(0, image.width - 1),
            step=1,
        )
        height_range = st.sidebar.slider(
            '縦',
            min_value=0,
            max_value=image.height - 1,
            value=(0, image.height - 1),
            step=1,
        )

        highlighted_image = image.copy()
        draw = ImageDraw.Draw(highlighted_image, 'RGBA')
        draw.rectangle(
            (0, 0, image.width - 1, height_range[0] - 1),
            fill=(0,) * 3 + (0x80,),
            width=0,
            outline=(0xFF,) * 4,
        )
        draw.rectangle(
            (0, height_range[0], width_range[0] - 1, height_range[1] - 1),
            fill=(0,) * 3 + (0x80,),
            width=0,
            outline=(0xFF,) * 4,
        )
        draw.rectangle(
            (width_range[1], height_range[0], image.width - 1, height_range[1] - 1),
            fill=(0,) * 3 + (0x80,),
            width=0,
            outline=(0xFF,) * 4,
        )
        draw.rectangle(
            (0, height_range[1], image.width - 1, image.height - 1),
            fill=(0,) * 3 + (0x80,),
            width=0,
            outline=(0xFF,) * 4,
        )
        st.image(highlighted_image, caption='Uploaded Image.', use_column_width=True)

        croped_image = image.crop(
            (width_range[0], height_range[0], width_range[1], height_range[1])
        ).convert('RGB')

        if not st.sidebar.checkbox('推論ストップ'):
            labels = predict(croped_image)

            df = pd.DataFrame(
                labels,
                columns=['id, name', 'prob.[%]'],
                index=range(1, len(labels) + 1),
            )
            df['prob.[%]'] *= 100
            df['cumulative prob.[%]'] = df['prob.[%]'].cumsum()
            df = df[df['prob.[%]'] > 1]

            fig = px.bar(
                x=df['id, name'],
                y=df['prob.[%]'],
                labels={'x': 'id, name', 'y': 'prob.[%]'},
                range_y=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

            st.table(df)
    return


if __name__ == "__main__":
    main()
