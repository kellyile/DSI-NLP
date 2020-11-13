from PIL import Image
import streamlit as st
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
import spacy_streamlit
import en_core_web_sm
import io
from typing import List, Optional
import markdown
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from plotly import express as px
import pydeck as pdk
import numpy as np
nlp = en_core_web_sm.load()



# matplotlib.use("TkAgg")
matplotlib.use("Agg")
COLOR = "black"
BACKGROUND_COLOR = "#fff"


menu = ["Home", "Benefits of NLP", "Toxic Comment Classifier", "Toxic Comment Classifier (multi-class)", "Get in Touch"]
side_bar_temp = """
                <style>
                .reportview-container .markdown-text-container {
                    font-family: monospace;
                }
                .sidebar .sidebar-content {
                    background-image: linear-gradient(#e3e3e3, #e3e3e3);
                    color: black;
                }
                .Widget>label {
                    color: black;
                    font-family: monospace;
                }
                [class^="st-b"]  {
                    color: black;
                    font-family: monospace;
                }
                .st-at {
                    background-color: white;
                }
                .st-dg {
                    background-color: #a6961f;
                }
                footer {
                    font-family: monospace;
                }
                .reportview-container .main footer, .reportview-container .main footer a {
                    color: black;
                }
                header .decoration {
                    background-image: none;
                }
                </style>
                """

safe_html = """ 
            <div style="background-color:#95ce1d;height:40px;width: 100%">
                <h2 style="color:white;text-align:center;"><b> Non-Toxic </b></h2>
            </div>
            """

danger_html = """  
              <div style="background-color:#ce1d1d;height:40px;width: 100%">
                <h2 style="color:white;text-align:center;"><b> Toxic </b></h2>
              </div>
              """

toxic_html = """  
              <div style="background-color:#ce1d1d;height:40px;width: 100%">
                <h2 style="color:white;text-align:center;"><b>Class 1: {}. Class 2: {}</b></h2>
              </div>
              """

white_space_html = """
                   <div style="background-color:white;height:40px;width: 100%"></div>
                   """

st.markdown(side_bar_temp, unsafe_allow_html=True,)
page = st.sidebar.markdown("""## Menu""")
page = st.sidebar.selectbox("", menu)


def select_block_container_style():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    st.sidebar.header("Block Container Style")
    max_width = 1200
    dark_theme = st.sidebar.checkbox("Dark Theme?", False)
    if dark_theme:
        global COLOR
        global BACKGROUND_COLOR
        BACKGROUND_COLOR = "rgb(17,17,17)"
        COLOR = "#fff"

    _set_block_container_style(
        max_width
    )


def _set_block_container_style(
        max_width: int = 1200,
        padding_top: int = 5,
        padding_right: int = 1,
        padding_left: int = 1,
        padding_bottom: int = 10,
):
    max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            {max_width_str}
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

select_block_container_style()


def add_resources_section():
    """Adds a resources section to the sidebar"""
    st.sidebar.header("Contact Us")
    st.sidebar.markdown(
        """
        - Email: <a href="kelly@dsi-program.com, nathanael@dsi-program.com, angelina@dsi-program.com">KAN@nlp-solutions.com</a>
        - Telephone: (021) 913 9119
        - Location [Google Maps] (http://maps.google.com/?q=-34.107210,18.470510)
""",
        unsafe_allow_html=True,
    )


add_resources_section()

if page == "Home":
    html_temp = """
                <div style="background-color:{};padding:10px">
                <h1 style="color:{};text-align:center;">KAN NLP SOLUTIONS</h1>
                </div>
                """
    st.markdown(html_temp.format("black", "#a6961f"), unsafe_allow_html=True)
    st.image("https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fbernardmarr%2Ffiles%2F2019%2F06%2F5-Amazing-Examples-Of-Natural-Language-Processing-NLP-In-Practice-1200x639.jpg", width=1169)
    st.image(Image.open(r"C:\Users\kile\Dropbox\Images\Home_page_pic.png"), width=1169)


if page == "Toxic Comment Classifier":
    html_temp = """
                <div style="background-color:{};padding:10px">
                <h1 style="color:{};text-align:center;">Toxic Comment Classifier</h1>
                </div>
                """
    st.markdown(html_temp.format("black", "#a6961f"), unsafe_allow_html=True)
    img = Image.open(r"C:\Users\kile\Dropbox\Images\wordcloud.png")
    st.image(img, width=1200)

    st.markdown("Social media is prolific in this modern age and has spurned countless interactions.\n"
                "While most of these are positive, some of them are abusive, insulting or even hate-based.\n"
                "It has thus become part of the mantle of social media organisations, to ensure that \n"
                "toxic conversation are identified, removed and prevented. This application aims to \n"
                "assist with this plight by identifying toxic and non toxic communication")

    comment = st.text_area("Enter Your Message")
    # drop_box = st.file_uploader("Upload your file.")
    bert_model_name = 'bert-base-cased'
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"


    class BertClassifier(nn.Module):

        def __init__(self, bert: BertModel, num_classes: int):
            super().__init__()
            self.bert = bert
            self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,

                    labels=None):
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask)
            cls_output = outputs[1]  # batch, hidden
            cls_output = self.classifier(cls_output)  # batch, 6
            cls_output = torch.sigmoid(cls_output)
            criterion = nn.BCELoss()
            loss = 0
            if labels is not None:
                loss = criterion(cls_output, labels)
            return loss, cls_output


    def predict_binary(input_string, model, predict_proba=False, threshold=.7):
        texts = []
        text = tokenizer.encode(input_string, add_special_tokens=True)
        if len(text) > 120:
            text = text[:119] + [tokenizer.sep_token_id]
        texts.append(torch.LongTensor(text))
        x = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            _, outputs = model(x, attention_mask=mask)
        prob_toxic = outputs.cpu().numpy()[0][0]
        # if predict_proba:
        #     st.markdown(prob_toxic)
        # else:
        if prob_toxic >= threshold:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)

        st.markdown(white_space_html, unsafe_allow_html=True)
        spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)

    if comment != "":
        docx = nlp(comment)
        path = r"https://uc9196a6b05d1bcc9ec612d5e302.dl.dropboxusercontent.com/cd/0/get/BC8uBmyBcj6sF8l_I39xF7enAlO3DeIm121lcHVxn5mOuOU3ze9AMTCiADHO6GNipaNwo-IcPYOnKMvU-PxlaEbXsM9B0J5UyCYV0N-C8q9gqgu9y4D-qb3734IwOhVUd7Y/file?_download_id=53183548221889719667987402181852167229972042290301559267877386361&_notify_doma"
        model = torch.hub.load_state_dict_from_url(path, map_location=torch.device('cpu'))
        predict_binary(comment, model, predict_proba=True)

if page == "Toxic Comment Classifier (multi-class)":
    html_temp = """
                <div style="background-color:{};padding:10px">
                <h1 style="color:{};text-align:center;">Toxic Comment Classifier</h1>
                </div>
                """
    st.markdown(html_temp.format("black", "#a6961f"), unsafe_allow_html=True)
    img = Image.open(r"C:\Users\kile\Dropbox\Images\wordcloud.png")
    st.image(img, width=1200)

    st.markdown("Social media is prolific in this modern age and has spurned countless interactions.\n"
                "While most of these are positive, some of them are abusive, insulting or even hate-based.\n"
                "It has thus become part of the mantle of social media organisations, to ensure that \n"
                "toxic conversation are identified, removed and prevented. This application aims to \n"
                "assist with this plight by identifying toxic and non toxic communication")

    comment = st.text_area("Enter Your Message")
    # drop_box = st.file_uploader("Upload your file.")
    bert_model_name = 'bert-base-cased'
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"


    class BertClassifier(nn.Module):

        def __init__(self, bert: BertModel, num_classes: int):
            super().__init__()
            self.bert = bert
            self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,

                    labels=None):
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask)
            cls_output = outputs[1]  # batch, hidden
            cls_output = self.classifier(cls_output)  # batch, 6
            cls_output = torch.sigmoid(cls_output)
            criterion = nn.BCELoss()
            loss = 0
            if labels is not None:
                loss = criterion(cls_output, labels)
            return loss, cls_output


    class BertClassifier(nn.Module):

        def __init__(self, bert: BertModel, num_classes: int):
            super().__init__()
            self.bert = bert
            self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,

                    labels=None):
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask)
            cls_output = outputs[1]  # batch, hidden
            cls_output = self.classifier(cls_output)  # batch, 6
            cls_output = torch.sigmoid(cls_output)
            criterion = nn.BCELoss()
            loss = 0
            if labels is not None:
                loss = criterion(cls_output, labels)
            return loss, cls_output

    def predict(text, model):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > 120:
            tokens = tokens[:119] + [tokens[-1]]
        x = torch.LongTensor(tokens)
        # y = torch.FloatTensor(row[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]])
        mask = (x != 0).float()
        with torch.no_grad():
            loss, outputs = model(x.unsqueeze(0), attention_mask=mask.unsqueeze(0))  # , labels=y)
        return outputs[0].numpy().tolist()

    if comment != "":
        docx = nlp(comment)
        bert_model_name = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        path = r"C:\Users\kile\Downloads\Models\model.pt"
        model = torch.load(path, map_location=torch.device('cpu'))
        model.eval()
        pred = predict(comment, model)
        class_index = np.argmax(pred)
        toxic_types = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        pred_df = pd.DataFrame(pred)
        pred_df.index = toxic_types
        pred_df.rename(columns={0: 'Class'}, inplace=True)
        toxic_class = pred_df.nlargest(2, ["Class"]).index[0]
        toxic_class2 = pred_df.nlargest(2, ["Class"]).index[1]
        st.markdown(toxic_html.format(toxic_class, toxic_class2), unsafe_allow_html=True)
        st.markdown(white_space_html, unsafe_allow_html=True)
        spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)
        chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        st.markdown("## Class probabilities")
        st.bar_chart(pred_df)
        # plt.barh(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], pred)

if page == "Benefits of NLP":

    html_temp = """
                <div style="background-color:{};padding:10px">
                <h1 style="color:{};text-align:center;">NLP for Business</h1>
                </div>
                """
    st.markdown(html_temp.format("black", "#a6961f"), unsafe_allow_html=True)

    class Cell:
        """A Cell can hold text, markdown, plots etc."""

        def __init__(
                self,
                class_: str = None,
                grid_column_start: Optional[int] = None,
                grid_column_end: Optional[int] = None,
                grid_row_start: Optional[int] = None,
                grid_row_end: Optional[int] = None,
        ):
            self.class_ = class_
            self.grid_column_start = grid_column_start
            self.grid_column_end = grid_column_end
            self.grid_row_start = grid_row_start
            self.grid_row_end = grid_row_end
            self.inner_html = ""

        def _to_style(self) -> str:
            return f"""
        .{self.class_} {{
            grid-column-start: {self.grid_column_start};
            grid-column-end: {self.grid_column_end};
            grid-row-start: {self.grid_row_start};
            grid-row-end: {self.grid_row_end};
        }}
        """

        def text(self, text: str = ""):
            self.inner_html = text

        def markdown(self, text):
            self.inner_html = markdown.markdown(text)

        def dataframe(self, dataframe: pd.DataFrame):
            self.inner_html = dataframe.to_html()

        def plotly_chart(self, fig):
            self.inner_html = f"""
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <body>
            <p>This should have been a plotly plot.
            But since *script* tags are removed when inserting MarkDown/ HTML i cannot get it to workto work.
            But I could potentially save to svg and insert that.</p>
            <div id='divPlotly'></div>
            <script>
                var plotly_data = {fig.to_json()}
                Plotly.react('divPlotly', plotly_data.data, plotly_data.layout);
            </script>
        </body>
        """

        def pyplot(self, fig=None, **kwargs):
            string_io = io.StringIO()
            plt.savefig(string_io, format="svg", fig=(2, 2))
            svg = string_io.getvalue()[215:]
            plt.close(fig)
            self.inner_html = '<div height="200px">' + svg + "</div>"

        def _to_html(self):
            return f"""<div class="box {self.class_}">{self.inner_html}</div>"""


    class Grid:
        """A (CSS) Grid"""

        def __init__(
                self,
                template_columns="1 1 1",
                gap="10px",
                background_color=COLOR,
                color=BACKGROUND_COLOR,
        ):
            self.template_columns = template_columns
            self.gap = gap
            self.background_color = background_color
            self.color = color
            self.cells: List[Cell] = []

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            st.markdown(self._get_grid_style(), unsafe_allow_html=True)
            st.markdown(self._get_cells_style(), unsafe_allow_html=True)
            st.markdown(self._get_cells_html(), unsafe_allow_html=True)

        def _get_grid_style(self):
            return f"""
        <style>
            .wrapper {{
            display: grid;
            grid-template-columns: {self.template_columns};
            grid-gap: {self.gap};
            background-color: {self.background_color};
            color: {self.color};
            }}
            .box {{
            background-color: {self.color};
            color: {self.background_color};
            border-radius: 5px;
            padding: 20px;
            font-size: 150%;
            }}
            table {{
                color: {self.color}
            }}
        </style>
        """

        def _get_cells_style(self):
            return (
                    "<style>"
                    + "\n".join([cell._to_style() for cell in self.cells])
                    + "</style>"
            )

        def _get_cells_html(self):
            return (
                    '<div class="wrapper">'
                    + "\n".join([cell._to_html() for cell in self.cells])
                    + "</div>"
            )

        def cell(
                self,
                class_: str = None,
                grid_column_start: Optional[int] = None,
                grid_column_end: Optional[int] = None,
                grid_row_start: Optional[int] = None,
                grid_row_end: Optional[int] = None,
        ):
            cell = Cell(
                class_=class_,
                grid_column_start=grid_column_start,
                grid_column_end=grid_column_end,
                grid_row_start=grid_row_start,
                grid_row_end=grid_row_end,
            )
            self.cells.append(cell)
            return cell


    @st.cache
    def get_dataframe() -> pd.DataFrame():
        """Dummy DataFrame"""
        data = [
            {"quantity": 1, "price": 2},
            {"quantity": 3, "price": 5},
            {"quantity": 4, "price": 8},
        ]
        return pd.DataFrame(data)


    def get_plotly_fig():
        """Dummy Plotly Plot"""
        return px.line(data_frame=get_dataframe(), x="quantity", y="price")


    def get_matplotlib_plt():
        get_dataframe().plot(kind="line", x="quantity", y="price", figsize=(5, 3))


    # def image(self, image_path, width="150px", height="200px", caption=""):
    #     data_uri = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    #     img_tag = '<br><figure ><center><img src="data:image/png;base64,{0}" width={2} height={3} ><figcaption>{1}</figcaption></center></figure>'.format(
    #         (data_uri, caption, width, height))
    #     self.inner_html = """<body>{0}</body>""".format(img_tag)


        # My preliminary idea of an API for generating a grid
    with Grid("1 1 1", color=COLOR, background_color=BACKGROUND_COLOR) as grid:
        grid.cell(
            class_="a",
            grid_column_start=2,
            grid_column_end=3,
            grid_row_start=1,
            grid_row_end=2,
        ).markdown("Faster response and decision making")
        grid.cell("b", 2, 3, 2, 3).text("Improved levels of engagement")
        grid.cell("c", 3, 4, 2, 3).text("Stronger Alignment with strategic objectives")
        grid.cell("d", 1, 2, 1, 3).text("Better engagement")
        grid.cell("e", 3, 4, 1, 2).markdown(
            "Better costumer understanding"
        )
        grid.cell("f", 1, 3, 3, 4).markdown(
            "# NLP SOLUTIONS"
        )
        grid.cell("g", 3, 4, 3, 4).text("Increased sales")

if page == "Get in Touch":
    html_temp = """
                <div style="background-color:{};padding:10px">
                <h1 style="color:{};text-align:center;">Contact Us</h1>
                </div>
                """
    st.markdown(html_temp.format("black", "#a6961f"), unsafe_allow_html=True)

    data = pd.DataFrame({
        'awesome cities': ['KAN NLP Solutions'],
        'lat': [-34.107210],
        'lon': [18.470510]
    })
    # st.pydeck_chart(
    #     viewport={
    #         'latitude': -34.107210,
    #         'longitude':  18.470510,
    #         'zoom': 4
    #             },
    #     layers=[{
    #         'type': 'ScatterplotLayer',
    #         'data': data,
    #         'radiusScale': 250,
    #         'radiusMinPixels': 5,
    #         'getFillColor': [248, 24, 148],
    #     }]
    # )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=-34.107210,
            longitude=18.470510,
            zoom=11,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
               'HexagonLayer',
               data=data,
               get_position='[lon, lat]',
               radius=200,
               elevation_scale=4,
               elevation_range=[0, 1000],
               pickable=True,
               extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))

    contact_imgs = [
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAMAAAAt85rTAAAAA1BMVEX///+nxBvIAAAASElEQVR4nO3BMQEAAADCoPVPbQo/oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICXAcTgAAG6EJuyAAAAAElFTkSuQmCC',
        'https://cdn4.iconfinder.com/data/icons/green-shopper/1049/email.png',
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAMAAAAt85rTAAAAA1BMVEX///+nxBvIAAAASElEQVR4nO3BMQEAAADCoPVPbQo/oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICXAcTgAAG6EJuyAAAAAElFTkSuQmCC',
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAMAAAAt85rTAAAAA1BMVEX///+nxBvIAAAASElEQVR4nO3BMQEAAADCoPVPbQo/oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICXAcTgAAG6EJuyAAAAAElFTkSuQmCC',
        'https://www.vippng.com/png/detail/79-790601_free-green-phone-icon.png'
    ]
    st.image(contact_imgs, width=190, caption=["", "Email", "", "", "Telephone"])
