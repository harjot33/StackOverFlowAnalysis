import json
import dash
import plotly.graph_objects as go
from dash.dependencies import Output, Input, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from badges import *
from posts import *
from tags import *
from users import *
from word_embedding import *
#from preprocessing import *
from io import BytesIO
import base64
from sentiment2 import sentimentplot2, sentimentplot1
from sentiment3 import sentimentplot3, sentimentplot4, sentimentplot5, sentimentplot6

tags_df = None
weekdays = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

years = {}
year = 2008
for i in range(2008, 2022):
    years[i] = str(year)
    year = year + 1


def load_app():

    global weekdays
    global years
    init_dataframe()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "18rem",
        "padding": "2rem 1rem",
        "background-color": "#343a40",
        "color": "#fff"
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "6rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

    sidebar = html.Div(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Div(children="COSMOS",
                                         style={'fontSize': 14, 'text-align': 'center', 'font-weight': 'bold'})),
                        dbc.Col(dbc.NavbarBrand("Stack Overflow Data Analysis", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                style={"textDecoration": "none"},
            ),
            html.Hr(),
            dbc.Nav(
                [
                    # Creating navigation links for side navbar
                    dbc.NavLink("Users", href="/", active="exact", className="nav-item"),
                    dbc.NavLink("Badges and Tags", href="/badges", active="exact", className="nav-item"),
                    dbc.NavLink("Posts", href="/posts", active="exact", className="nav-item"),
                    dbc.NavLink("Comments", href="/comments", active="exact", className="nav-item"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
        id="sidebar"
    )

    # Dynamic content loading with navigation
    content = html.Div(
        id="page-content",
        children=[],
        style=CONTENT_STYLE)

    # Card for prediction
    card = dbc.Card(
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col([
                        html.H5("Select technology or tags associated with the question"),
                        dbc.FormGroup([
                            dcc.Dropdown(id="tag_dropdown", value='',
                                         options=[{"label": key, "value": key} for key in tags_df["tags"].unique()]),
                        ]),
                    ]),
                    dbc.Col([
                        html.H5("Select the day of week when the question is intended to be posted"),
                        dbc.FormGroup([
                            dcc.Dropdown(id="question_weekday_dropdown", value='',
                                         options=[{"label": key, "value": key} for key in weekdays.keys()]),
                        ]),
                    ]),
                ]),
            html.Div([
                dbc.Button('Predict', id='predict_btn', color='primary', block=True),
            ], className="d-grid gap-2 col-6 mx-auto"),
            html.Br(),
            html.Div(id="predict")
        ]),
        outline=True
    )

    # Contains data shown on users screen
    user = html.Div([
        dcc.Graph(id="users_map"),
        html.Div([
            dbc.Button('Refresh', id='refresh_btn', color='primary', block=True),
        ], className="d-grid gap-2 col-6 mx-auto"),
        html.Br(),
        html.Br(),
        dbc.FormGroup([
            dbc.Label("Top 1000 user based on:"),
            dcc.Dropdown(id="dropdown_dataset", value=1,
                         options=[
                             {"label": "Reputation", "value": 1},
                             {"label": "Up-votes", "value": 2}
                         ]),
        ]),
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id='graph')),
            ])
        ], style={"height": "35rem",
                  "width": "60rem",
                  "left": "0",
                  "right": "0",
                  "position": "center",
                  "position": "relative",
                
                  "margin-right": "0rem",
                  "padding": "2rem 1rem",
                  }),

        html.Div([
            dbc.Card(
                dbc.CardBody(id='selectData', children=[
                    html.Div([html.H4("Name: "), html.H4(id='name')]),
                    html.Div([html.H4("ID: "), html.P(id='id')]),
                    html.Div([html.H4("Location: "), html.P(id='location')]),
                    html.Div([html.H4("Date of Joining : "), html.P(id='date')]),
                    html.Div([html.H4("Reputation: "), html.P(id='reputation')]),
                    html.Div([html.H4("Up-votes: "), html.P(id='upvote')]),
                ]),
            )
        ], style={'display': 'block'}, id='display_div'),

        html.Br(),
        html.H4("Top 10 countries based on reputation"),
        html.Div([
            dcc.Graph(figure=generate_user_reputation()),
        ]),

    ])

    # Contains data shown on badges screen
    badges = html.Div([
        html.H4("Total upvotes based on country"),
        html.Div([
            dcc.Graph(id="upvotes_chart", figure=generate_user_upvotes())
        ]),
        html.Br(),
        html.Div([
            dcc.Graph(figure=generate_badges_sunburst_chart()),
        ]),

        html.Br(),
        html.Div([
            dcc.Graph(figure=generate_bar_chart_badges()),
        ]),
        html.Div([
            html.H4("Frequently used Tags"),
        ]),
        html.Br(),
        html.Img(id="tags_wordcloud", style={'height': '80%', 'width': '100%'}),
        html.Br()
    ])

    # Contains data shown on posts screen
    posts = html.Div([
        html.H4("Total Post question answer in each years"),
        html.Br(),
        html.Br(),
        html.Div([
            dcc.Graph(figure=generate_table()),
        ]),
        html.Br(),
        html.H4("Day of week analysis"),
        html.Br(),
        html.Br(),
        dcc.Graph(figure=generate_post_bar_chart()),
        html.Br(),
        html.H4("Word similarity based on post content"),
        html.Br(),
        html.Br(),
        dcc.Graph(id="vector_fig", figure=vector_fig()),
        html.Br(),
        html.H4("Answer prediction based on tags and the weekday when a question was (or can be) posted"),
        html.Br(),
        card,
        html.Br(),
        html.Br(),
       
        
        # html.Br(),
        # dcc.Graph(id="posts"),
        # html.Div([
        #     dbc.Button('Refresh', id='refresh', color='primary', block=True),
        # ], className="d-grid gap-2 col-6 mx-auto"),
        html.Br(),
        html.Br(),
        html.H1(children='Question Posted on year Visualization'),
        html.Hr(),
        html.Div([dcc.Graph(id='post_tags')]),
        html.Hr(),
        dbc.Label("select year"),
        html.Div([
            dcc.Slider(
                id='my-slider',
                min=2013,
                max=2018,
                dots=True,
                value=2013,
                marks={2012: {'label': '2012'},
                       2013: {'label': '2013'},
                       2014: {'label': '2014'},
                       2015: {'label': '2015'},
                       2016: {'label': '2016'},
                       2017: {'label': '2017'},
                       2018: {'label': '2018'},
                       2019: {'label': '2019'},
                       2020: {'label': '2020'},
                       },
                included=False),

            html.Div(id='slider-output-container1')
        ]),
        
        html.Br(),
        html.Br(),
        html.Div([
            #dcc.Graph(id="post_table", figure=generate_table()),
        ]),

        html.Div([
            html.H5("Select technology or tags to find its trend over the years"),
            dbc.FormGroup([
                dcc.Dropdown(id="tag_trend_dropdown", value='java',
                             options=[{"label": key, "value": key} for key in tags_df["tags"].unique()]),
            ]),
            dcc.Graph(id="tag_trend")
        ]),
    ])

    # Contains data shown on comments screen
    comments = html.Div([
        html.H1(children="Sentimental Analysis"),
        html.Div(children='User Questions - Answers - Comments - Programming Languages Sentiment Study'),
        html.Hr(),

        # Creating Div which displays the information regarding what kind of sentimental analysis will be there.
        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("User Questions Sentiment Analysis"),
                ]),
                style={"width": "18rem"},
            ),
        ]),
        # Dropdown for the user to choose the programming language
        dbc.Label('Choose programming languages from below to see how sentiments differ in questions.'),
        dcc.Dropdown(id="planalysis",
                     options=[
                         {'label': 'Java', 'value': 'java'},
                         {'label': 'Javascript', 'value': 'javascript'},
                         {'label': 'Python', 'value': 'python'},
                         {'label': 'C#', 'value': 'c#'},
                         {'label': 'PHP', 'value': 'php'},
                         {'label': 'Android', 'value': 'android'},
                         {'label': 'C++', 'value': 'c++'},
                         {'label': 'IOS', 'value': 'ios'},
                         {'label': 'SQL', 'value': 'sql'},
                         {'label': 'HTML', 'value': 'html'},
                         {'label': 'R', 'value': 'r'},
                         {'label': 'JQuery', 'value': 'jquery'},
                         {'label': 'C', 'value': 'c'},
                         {'label': 'MySQL', 'value': 'mysql'},
                         {'label': 'Ruby On Rails', 'value': 'ruby-on-rails'},
                         {'label': 'CSS', 'value': 'css'},
                         {'label': 'Node JS', 'value': 'node.js'},
                         {'label': 'Excel', 'value': 'excel'},
                         {'label': 'Angular', 'value': 'angular'},
                     ],
                     multi=True,
                     value=["java", "python"]
                     ),
        # Run Analysis button runs the analysis on the chosen programming languages and then a chart is displayed
        dbc.Button('Run Analysis', id='planalysisbutton', color='primary', style={'margin-bottom': '1em'}, block=True,
                   n_clicks=0),
        dbc.Row([
            dbc.Col(dcc.Graph(id='firstvisualization')),
        ]),

        # Dropdown for the user to choose the programming language
        dbc.FormGroup([
            dbc.Label('Please select the programming languages to see their respective Sentiment Ratios'),
            dcc.Dropdown(id="planalysis2",
                         options=[
                             {'label': 'Java', 'value': 'java'},
                             {'label': 'Javascript', 'value': 'javascript'},
                             {'label': 'Python', 'value': 'python'},
                             {'label': 'C#', 'value': 'c#'},
                             {'label': 'PHP', 'value': 'php'},
                             {'label': 'Android', 'value': 'android'},
                             {'label': 'C++', 'value': 'c++'},
                             {'label': 'IOS', 'value': 'ios'},
                             {'label': 'SQL', 'value': 'sql'},
                             {'label': 'HTML', 'value': 'html'},
                             {'label': 'R', 'value': 'r'},
                             {'label': 'JQuery', 'value': 'jquery'},
                             {'label': 'C', 'value': 'c'},
                             {'label': 'MySQL', 'value': 'mysql'},
                             {'label': 'Ruby On Rails', 'value': 'ruby-on-rails'},
                             {'label': 'CSS', 'value': 'css'},
                             {'label': 'Node JS', 'value': 'node.js'},
                             {'label': 'Excel', 'value': 'excel'},
                             {'label': 'Angular', 'value': 'angular'},
                         ],
                         multi=True,
                         value=["javascript", "php", "html"]
                         )
        ]),
        # Run Analysis button runs the analysis on the chosen programming languages and then a chart is displayed
        dbc.Button('Discover Sentiment Ratios', id='planalysisbutton2', color='primary', style={'margin-bottom': '1em'},
                   block=True, n_clicks=0),

        dbc.Row([
            dbc.Col(dcc.Graph(id='secondvisualization')),
        ]),
        # Card showing information regarding User Answer Analysis
        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("User Answers Sentiment Analysis"),
                ]),
                style={"width": "18rem"},
            ),
        ]),
        # Slider for choosing the desired year to see the chart.
        dbc.Label('Choose the year to see an annual summary of the sentiments in user answers.'),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=2008, max=2021, step=1, value=2008, marks=years),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='thirdvisualization')),
        ]),

        # Slider for choosing the desired year to see the chart.
        dbc.Label('Choose the year to see the comparison of positive and negative answers.'),
        dbc.FormGroup([
            dbc.Label(id='slider-value2'),
            dcc.Slider(id="slider2", min=2008, max=2021, step=1, value=2008, marks=years),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='fourthvisualization')),
        ]),

        # Dropdown for choosing ths specific year for the user.
        dbc.Label('Sentiment Based User Score Analysis'),
        dbc.FormGroup([
            dbc.Label('Choose Year 1'),
            dcc.Dropdown(id='year1', value=2008,
                         options=[{'label': '2008', 'value': 2008},
                                  {'label': '2009', 'value': 2009},
                                  {'label': '2010', 'value': 2010},
                                  {'label': '2011', 'value': 2011},
                                  {'label': '2012', 'value': 2012},
                                  {'label': '2013', 'value': 2013},
                                  {'label': '2014', 'value': 2014},
                                  {'label': '2015', 'value': 2015},
                                  {'label': '2016', 'value': 2016},
                                  {'label': '2017', 'value': 2017},
                                  {'label': '2018', 'value': 2018},
                                  {'label': '2019', 'value': 2019},
                                  {'label': '2020', 'value': 2020},
                                  {'label': '2021', 'value': 2021}
                                  ], clearable=False,
                         searchable=False),
        ]),

        dbc.FormGroup([
            dbc.Label('Choose Year 2'),
            dcc.Dropdown(id='year2',
                         value=2009,
                         options=[{'label': '2008', 'value': 2008},
                                  {'label': '2009', 'value': 2009},
                                  {'label': '2010', 'value': 2010},
                                  {'label': '2011', 'value': 2011},
                                  {'label': '2012', 'value': 2012},
                                  {'label': '2013', 'value': 2013},
                                  {'label': '2014', 'value': 2014},
                                  {'label': '2015', 'value': 2015},
                                  {'label': '2016', 'value': 2016},
                                  {'label': '2017', 'value': 2017},
                                  {'label': '2018', 'value': 2018},
                                  {'label': '2019', 'value': 2019},
                                  {'label': '2020', 'value': 2020},
                                  {'label': '2021', 'value': 2021}
                                  ], clearable=False,
                         searchable=False),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='fifthvisualization')),
        ]),

        # This portion is for the Sentimental Analysis of User Comments
        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("User Comments Sentimental Analysis"),
                ]),
                style={"width": "18rem"},
            ),
        ]),
        dbc.Label('Choose the year to see an annual summary of the sentiments in user comments.'),
        dbc.FormGroup([
            dbc.Label(id='slider-value3'),
            dcc.Slider(id="slider3", min=2008, max=2021, step=1, value=2008, marks=years),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='sixthvisualization')),
        ]),
    ])

    app.layout = html.Div([
        dcc.Location(id="url"),
        dbc.Row(
            children=[
                dbc.Col(
                    children=html.Div([
                        dcc.Store(id='side_click'),
                        sidebar,
                    ]), width=2
                ),
                dbc.Col(
                    children=content, width=10
                )
            ]
        )
    ])

    # app callback function to navigate through the side navbar
    @app.callback(
        Output("page-content", "children"),
        [Input("url", "pathname")]
    )
    def route(pathname):
        if pathname == "/":
            return user
        elif pathname == "/badges":
            return badges
        elif pathname == "/posts":
            return posts
        elif pathname == "/comments":
            return comments

    # app callback function to load user geolocation map
    @app.callback(
        [Output('users_map', 'figure'),
         Output('refresh_btn', 'n_clicks')],
        [Input('users_map', 'clickData')],
        [Input('refresh_btn', 'n_clicks')]
    )
    def update_figure(users_map, btn_click):
        if users_map is None or btn_click is not None:
            fig = load_user_by_country()
            btn_click = None
        else:
            fig = active_users_per_year(users_map["points"][0]["location"])
        return fig, btn_click

    # app callback function for wordcloud base don tags
    @app.callback(
        Output('tags_wordcloud', 'src'),
        [Input('tags_wordcloud', 'id')]
    )
    def update_wordcloud(elem_id):
        img = BytesIO()
        generate_word_cloud().save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

   

    @app.callback(
        [Output("posts", "figure"),
         Output('refresh', 'n_clicks')],
        [Input("posts", 'clickData')],
        [Input("refresh", 'n_clicks')]
    )
    def update_post_per_year_mon(posts, btn_click):

        print(posts)

        if posts is None or btn_click is not None:
            fig = generate_line_graph_post_per_year()
            btn_click = None
        else:
            print(posts['points'][0]['x'])
            year = posts['points'][0]['x']
            fig = generate_line_graph_post_per_month(year)

        return fig, btn_click

    @app.callback(
        Output('post_tags', 'figure'),
        Input('my-slider', 'value')
    )
    def update_figure(year_value):
    
        fig = generate_tag_years(year_value)
        return fig

    @app.callback(
        Output('graph', 'figure'),
        [Input('dropdown_dataset', 'value')]
    )
    def update_figure(dropdown_dataset):

        if dropdown_dataset == 1:
            return generate_tree_map_reputation()
        elif dropdown_dataset == 2:
            return generate_tree_map_upvotes()

    @app.callback(
        Output('display_div', 'style'),
        Output('name', 'children'),
        Output('id', 'children'),
        Output('location', 'children'),
        Output('date', 'children'),
        Output('reputation', 'children'),
        Output('upvote', 'children'),
        Input('graph', 'hoverData')
    )
    def display_selected_data(selectedData):

        # print(selectedData)
        l = []
        if selectedData is not None:
            l = selectedData['points'][0]['currentPath'].split("/")
            print(l)
        if len(l) == 4:
            d = dataf()
            display_name = l[2]
            mask = d['display_name'] == display_name
            df_ = d[mask]
            name = df_.values[0][1]
            id = df_.values[0][0]
            loc = df_.values[0][4]
            date = df_.values[0][2]
            dateofjoin = str(date).split(" ")[0]
            repu = df_.values[0][5]
            up_vote = df_.values[0][6]
            return {'display': 'block'}, name, id, loc, dateofjoin, repu, up_vote
        else:
            return {'display': 'none'}, "", "", "", "", "", ""

    # app callback function to fetch the prediction for response time based on tags.
    @app.callback(
        Output('predict', 'children'),
        [State('tag_dropdown', 'value')],
        [State('question_weekday_dropdown', 'value')],
        [Input('predict_btn', 'n_clicks')]
    )
    def update_prediction(tag_dropdown, question_weekday_dropdown, predict_btn):
        global weekdays
        predicted_value = predict(tag_dropdown, weekdays.get(question_weekday_dropdown))
        
        if predicted_value > 0:
            return "The predicted response time for a question posted on {} with {} tag is {} hour(s)".format(
                question_weekday_dropdown, tag_dropdown, predicted_value)
        else:
            return "The predicted response time for a question posted on {} with {} tag is {} hour(s)".format(
                question_weekday_dropdown, tag_dropdown, "Undefined")

    # app callback function to show trends in tag over the years
    @app.callback(
        Output('tag_trend', 'figure'),
        [Input('tag_trend_dropdown', 'value')]
    )
    def update_tag_trend(dropdown_value):
        fig = generate_tag_trend(dropdown_value)
        return fig

    # First Callback -
    @app.callback(
        Output(component_id='firstvisualization', component_property='figure'),
        [State(component_id='planalysis', component_property='value'), ],
        [Input(component_id='planalysisbutton', component_property='n_clicks')]
    )
    def func(planalysis, n_clicks):
        fig = sentimentplot1(planalysis)
        return fig

    # Second Callback
    @app.callback(
        Output(component_id='secondvisualization', component_property='figure'),
        [State(component_id='planalysis2', component_property='value'), ],
        [Input(component_id='planalysisbutton2', component_property='n_clicks')]
    )
    def func2(planalysis2, n_clicks):
        fig = sentimentplot2(planalysis2)
        return fig

    # Third Callback
    @app.callback(
        Output(component_id='thirdvisualization', component_property='figure'),
        [Input(component_id='slider', component_property='value')],
    )
    def func3(slidervalue):
        fig = sentimentplot3(slidervalue)
        return fig

    # Fourth Callback
    @app.callback(
        Output(component_id='fourthvisualization', component_property='figure'),
        [Input(component_id='slider2', component_property='value')],
    )
    def func4(slidervalue2):
        fig = sentimentplot4(slidervalue2)
        return fig

    # Fifth Callback
    @app.callback(
        Output(component_id='fifthvisualization', component_property='figure'),
        [Input(component_id='year1', component_property='value'),
         Input(component_id='year2', component_property='value')],
    )
    def func5(year1, year2):
        fig = sentimentplot5(year1, year2)
        return fig

    # Sixth Callback
    @app.callback(
        Output(component_id='sixthvisualization', component_property='figure'),
        [Input(component_id='slider3', component_property='value')],
    )
    def func4(slidervalue3):
        fig = sentimentplot6(slidervalue3)
        return fig

    return app


def init_dataframe():
    global tags_df
    users_init()
    tags_df = pd.read_csv(Path('./Data/Processed/', 'tags_modified_encoded.csv'))
    tags_df = tags_df.fillna(0)
    tags_df = tags_df.groupby("tags").size().reset_index(name='counts')
    tags_df = tags_df.sort_values(by="counts", ascending=False)
    tags_df = tags_df[:20]
    tags_init()
    badges_init()
    posts_init()
    train_model_for_prediction()


if __name__ == "__main__":
    app_run = load_app()
    app_run.run_server(debug=True)
