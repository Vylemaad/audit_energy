#######  Usefull Imports  ########
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from math import *
from altair import datum
from random import randint

#######  Usefull global variable  ########
wrong_filled_products = dict()

#Rename usage type for a more pertinent granularity :
rename_usetype = {'SPS-District K-12' : 'K-12 School',
                  'Residence Hall' : 'Residence Hall/Dormitory',
                  'Supermarket / Grocery Store' : 'Supermarket/Grocery Store',
                  'University' : 'College/University',
                  'Senior Care Community' : 'Residential Care Facility',
                  'Hospital (General Medical & Surgical)' : 'Hospital',
                  'Other/Specialty Hospital' : 'Hospital',
                  'Outpatient Rehabilitation/Physical Therapy' : 'Medical Office',
                  'Urgent Care/Clinic/Other Outpatient' : 'Medical Office',
                  'Adult Education' : 'Education',
                  'Vocational School' : 'Education',
                  'Fast Food Restaurant' : 'Restaurant',
                  'Convenience Store without Gas Station' : 'Retail Store',
                  'Food Sales' : 'Food Service',
                  'Convention Center' : 'Social/Meeting Hall',
                  'Multifamily Housing' : 'Lodging/Residential',
                  'Personal Services (Health/Beauty, Dry Cleaning, etc)' : "Services",
                  'Repair Services (Vehicle, Shoe, Locksmith, etc)' : "Services",
                  'Self-Storage Facility' : 'Warehouse',
                  'Other - Lodging/Residential' : 'Lodging/Residential',
                  'Other - Recreation' : 'Recreation',
                  'Other - Education' : 'Education',
                  'Other - Mall' : 'Mall',
                  'Other - Public Services' : "Public Services",
                  'Other - Utility' : "Services",
                  'Other - Services' : "Services",
                  'Other - Technology/Science' : 'Technology/Science',
                  'Other - Restaurant/Bar' : 'Restaurant',
                  'Other - Entertainment/Public Assembly' : 'Social/Meeting Hall'}

list_of_color =['cadetblue', 'lightcoral', 'mediumvioletred', 'lavender', 'lawngreen', 'lemonchiffon', 'tomato', 'darkblue',
                'deeppink', 'saddlebrown','chocolate','dodgerblue','forestgreen',
                'darkgoldenrod', 'darkyellow','darkgreen', 'darkkhaki', 'darkmagenta',
                 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
                 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray',
                 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold',
                 'goldenrod', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki',
                 'lightblue', 'lightcyan', 'orange', 'burlywood',
                 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
                 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen',
                 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'cyan', 'teal', 
                 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream',
                 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod',
                 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
                 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
                 'sienna', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan',
                 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple', 'cornflowerblue']

    
#######  Usefull global functions  ########
def is_a_number(myVariable):
    if type(myVariable) == np.float64 or type(myVariable) == int \
    or type(myVariable) == float or (type(myVariable) == str and myVariable.isnumeric()):
        return True
    elif type(myVariable) == str:
        if len(myVariable) > 1 and myVariable[0]=="-":
            return is_a_number(myVariable[1:])
        L = myVariable.split(".")
        if len(L)>2 and (not L[0].isnumeric()):
            print("A",myVariable, len(L), L[0])
            return False
        myVariable = L[-1]
        if (type(myVariable) == str and myVariable.isnumeric()):
            return True
        L = myVariable.split("e-")
        return len(L)==2 and L[0].isnumeric() and  L[1].isnumeric()
    else:
        print(type(myVariable))
        return False


def is_approx_to(value, wanted_value, approx):
    return value <= wanted_value*(1.0+approx) and value >= wanted_value*(1.0-approx)


def show_corr_matrix(corr):
    
    chart_corr = (corr.stack()
                  .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
                  .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
    chart_corr['correlation_label'] = chart_corr['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
    base = alt.Chart(chart_corr).encode(
        x='variable2:O',
        y='variable:O'    
    )

    # Text layer with correlation labels
    # Colors are for easier readability
    text = base.mark_text().encode(
        text='correlation_label',
        color=alt.condition(
            alt.datum.correlation > 0.5, 
            alt.value('white'),
            alt.value('black')
        )
    )

    # The correlation heatmap itself
    cor_plot = base.mark_rect().encode(
        color='correlation:Q'
    )

    (cor_plot + text).properties(width = 400, height = 400).display()
    
    
def is_normalized(dataset, field):
    shapiro_b = True
    Agostino_Pearson_b = False
    Anderson_Darling_b = True
    data = dataset[field]
    if type_of_data(dataset, field) == "categorical data" :
        return "ERROR : no normalized test on categorical data !"

    #Shapiro test
    #p-value may not be accurate for N > 5000.
    if data.shape[0] > 5000 : 
        number_of_bins = int(data.shape[0]/5000)
        for binned_data in pd.cut(data, bins=number_of_bins):
            stat, p = shapiro(binned_data)

            # interpret
            alpha = 0.05
            if p <= alpha:
                shapiro_b = False
                break
    else :
        stat, p = shapiro(data)
        # interpret
        alpha = 0.05
        if p <= alpha:
            shapiro_b = False

    # Agostino & Pearson test
    stat, p = normaltest(data)
    # interpret
    alpha = 0.05
    if p > alpha:
        Agostino_Pearson_b = True

    # Anderson-Darling Test
    result = anderson(data)
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic >= result.critical_values[i]:
            Anderson_Darling_b = False
            break
    return Anderson_Darling_b and shapiro_b and Agostino_Pearson_b

def type_of_data(dataset, field):
    qualitatifs_types = ['category','object','bool','boolean']
    type_field = dataset[field].dtype
    if type_field in qualitatifs_types :
        return "nominal"
    else:
        return "quantitative"

def analyse_univarie(dataset, field, compare_fields, etape_name):
    #Recuperation des informations importantes qui seront utilisée
    describe = dataset[field].describe()
    count_value = dataset[field].value_counts()
    perc_null = round((dataset[field].isnull().sum() / dataset.shape[0] )* 100,1)
    
    #Titre du graphe
    graph_title = [f'Catégorie {field}']
    if perc_null == 0 :
        graph_title.append('(Aucune donnée nulle)')
    else:
        graph_title.append(f"({perc_null}% de données nulles non représentées)")
        
    #Check variable type : numeric or categoric
    type_var_cible = type_of_data(dataset, field)
    if type_var_cible == "quantitative": # numerique variable
        
        # Graph principal valeurs/fréquence
        bar = alt.Chart(dataset.dropna(subset = [field])).transform_joinaggregate(total=f'count({field})').transform_calculate(
              pct='1 / datum.total').mark_bar(color=list_of_color[0], invalid = "filter").encode(x = alt.X(field, type=type_var_cible),
                                                                         y = alt.Y('sum(pct):Q',
                                                                                   axis=alt.Axis(format='%',
                                                                                title='Fréquence des valeurs')))

        # Affichage de la median et de la moyenne
        characs = {"values" : [np.nanmedian(dataset[field]), describe["mean"]], 
                   "Data Characteristics" : ["Mediane", "Moyenne"]}
        data_charac = alt.Chart(pd.DataFrame(characs)).mark_rule(strokeWidth=2).encode(x=alt.X(f"values:Q",
                                                                                  axis=alt.Axis(
                                                                                  title=f"Valeurs de la catégorie {field}")
                                                                                 ), 
                                                                          color = alt.Color( "Data Characteristics:N",
                                                                                            legend=alt.Legend(columns=2,
                                                                                                              direction="vertical",
                                                                                                              orient="bottom")))

        # Affichage des valeurs min et max
        annot_min = alt.Chart(pd.DataFrame({"valeur":[describe['min']],"where" : [0]})).mark_text(
                                                                align='center',
                                                                baseline='line-top',
                                                                fontSize = 15,
                                                                dy = -270,
                                                                dx = -125
                                                                ).encode(y = "where:Q", text = "label:N"
                                                                        ).transform_calculate(label=f"'min = '+{datum.valeur}")
        annot_max = alt.Chart(pd.DataFrame({"valeur":[describe['max']],"where" : [0]})).mark_text(
                                                                align='center',
                                                                baseline='line-top',
                                                                fontSize = 15,
                                                                dy = -270,
                                                                dx = 125
                                                                ).encode(y = "where:Q", text = "label:N"
                                                                        ).transform_calculate(label=f"'max = '+{datum.valeur}")
        
        display = alt.layer(bar + data_charac + annot_min + annot_max).properties(title=graph_title, width=330, height=250)

    elif "unique" in describe.keys() and describe['unique'] < 15 : # variable categorique discrete

        bar = alt.Chart(dataset.dropna(subset = [field])).transform_joinaggregate(total=f'count({field})').transform_calculate(
              pct='1 / datum.total')
        bar = bar.mark_arc(invalid = "filter").encode(
    theta=alt.Theta(field='pct', aggregate='sum', type="quantitative"),
    color=alt.Color(field=field, type="nominal", legend=alt.Legend(columns=2, direction="vertical", orient="bottom")))
        display = bar.properties(title=graph_title, width=400, height=300)    
            
    else :
        print(describe)
        return
    list_of_chart = []
    subset = compare_fields + [field]
    bar = alt.Chart(dataset.dropna(subset = subset))
    
    for i,f in enumerate(compare_fields) :
        type_var = type_of_data(dataset, f)
        if (type_var == "nominal") ^ (type_var_cible == "nominal"):
            if type_var == "nominal":
                graph = bar.mark_point().encode(x=alt.X(f,type = type_var, axis=alt.Axis(labels=False)),
                                                color = f,
                                                y=alt.Y(field, type = type_var_cible)
                                               ).properties(width = 185, height = 185)
            else:
                graph = bar.mark_point().encode(x=alt.X(f,type = type_var),
                                                color = alt.Color(field, legend=None),
                                                y=alt.Y(field, type = type_var_cible, axis=alt.Axis(labels=False))
                                                ).properties(width = 185, height = 185)
            list_of_chart.append(graph)
        else:
            colo = list_of_color[i+1]
            graph = bar.mark_point(opacity=0.2, color=colo).encode(alt.X(f, type = type_var),
                                    alt.Y(field, type = type_var_cible)).properties(width = 185, height = 185)
            list_of_chart.append(graph)
    
    #stack charts
    if len(list_of_chart) > 1 :
        chart = list_of_chart[0]|list_of_chart[1]
        for i in range(2,len(list_of_chart),2):
            if i != len(list_of_chart)-1 :
                chart = chart & (list_of_chart[i]|list_of_chart[i+1])
            else:
                chart = chart | list_of_chart[i]
        chart = chart.properties(title =alt.TitleParams( f"Interaction avec les autres variables:",anchor='middle', align='center'))
        display = (display | chart).properties(title=alt.TitleParams([f"Analyse et descriptive Univariée de",
                                                                      f"{field} sur {etape_name}"],
                                                                     anchor='middle',
                                                                     align='center')).resolve_legend(color='independent')
            
    elif  len(list_of_chart) == 1:
        chart = list_of_chart[0].properties(title=alt.TitleParams( f"Interaction avec les autres variables:",anchor='middle', align='center'))

        display = (display | chart ).properties(title =alt.TitleParams( text=f"Analyse descriptive et univariée de",
                                                                       subtitle=f"{field} sur {etape_name}",
                                                                       anchor='middle',
                                                                       align='center')).resolve_legend(color='independent')
    display.display()
    
def display_scree(scree) :

    eboulis = pd.DataFrame([range(1,len(scree)+1), scree, np.cumsum(scree)],
                           index = ['index','perc_inertie', 'Cumulative']).transpose()
    
    base = alt.Chart(eboulis)
    
    bar = base.mark_bar(size=35).encode(x = alt.X('index', axis=alt.Axis(tickMinStep=1, title="Rang de l'axe d'inertie")),
                                        y = alt.Y("perc_inertie", axis=alt.Axis(format='%', title="Pourcentage d'inertie")))
    
    line =  base.mark_line(color='red', point=alt.OverlayMarkDef(color="red")).encode(x = "index",
                                                                                      y="Cumulative",
                                                                                      tooltip=[alt.Tooltip(title="% cumulé",
                                                                                                           field="Cumulative",
                                                                                                           format="%")])
                                                                                                           
                                                                                                           
    legendes = pd.DataFrame({'legende' : ["Pourcentages Cumulés","Pourcentages Cumulés"],
                            'x':[0,0], 'y': [0,0]})
    scale = alt.Scale(domain=["Pourcentages Cumulés"], range= ["red"])
    legend= alt.Chart(legendes).mark_line().encode(x=alt.X('x:Q', axis=alt.Axis(labels=False, title="")),
                                                   color=alt.Color('legende', scale=scale, title=""),
                                                   y=alt.Y('y:Q', axis=alt.Axis(labels=False, title="")))



    (bar+line+legend).properties(title='Eboulis des valeurs propres').display()
    
def display_corr_circle(scree, pcs, plan_fact, labels):

    # affichage du cercle et des lignes horizontales et verticales
    demi_circle_y = np.arange(1.001,step=0.005)
    demi_circle_x = np.array([sqrt(1-pow(y,2)) for y in demi_circle_y])
    demi_circle_x = np.append(demi_circle_x*-1, demi_circle_x)
    demi_circle_y = np.append(demi_circle_y,demi_circle_y)

    dict_haut_verti = dict()
    dict_haut_verti["x"] = np.append(demi_circle_x, [0,0])
    dict_haut_verti["y"] = np.append(demi_circle_y, [-1, 1])
    dict_haut_verti["mode"] = np.append(["a-line"]*len(demi_circle_y), ["dash","dash"])

    dict_bas_hori = dict()
    dict_bas_hori["x"] = np.append(demi_circle_x, [-1,1])
    dict_bas_hori["y"] = np.append(demi_circle_y*-1, [0, 0])
    dict_bas_hori["mode"] = np.append(["a-line"]*len(demi_circle_y), ["dash","dash"])

    dash_verti = alt.Chart(pd.DataFrame(dict_haut_verti)).mark_line(color='grey').encode(x = alt.X('x', type='quantitative', scale=alt.Scale(domain=[-1, 1]),
                                            axis=alt.Axis(title=f"Composante n°{plan_fact[0]+1} ({round(scree[plan_fact[0]]*100)}%)")),
                                            y = alt.Y("y", type='quantitative',scale=alt.Scale(domain=[-1, 1]),
                                            axis=alt.Axis(title=f"Composante n°{plan_fact[1]+1} ({round(scree[plan_fact[1]]*100)}%)"))
                                            , strokeDash=alt.StrokeDash('mode', legend=None))

    dash_hori = alt.Chart(pd.DataFrame(dict_bas_hori)).mark_line(color='grey').encode(x = alt.X('x', type='quantitative', scale=alt.Scale(domain=[-1, 1]),
                                            axis=alt.Axis(title=f"Composante n°{plan_fact[0]+1} ({round(scree[plan_fact[0]]*100)}%)")),
                                            y = alt.Y("y", type='quantitative',scale=alt.Scale(domain=[-1, 1]),
                                            axis=alt.Axis(title=f"Composante n°{plan_fact[1]+1} ({round(scree[plan_fact[1]]*100)}%)"))
                                            , strokeDash=alt.StrokeDash('mode', legend=None))


    dict_axes = {k:[] for k in ["x","y","axis name:"]}
    for i, (x,y) in enumerate(pcs[plan_fact].T):
        dict_axes["x"].extend([0,x])
        dict_axes["y"].extend([0,y])
        dict_axes["axis name:"].extend([labels[i]]*2)

    axes = alt.Chart(pd.DataFrame(dict_axes)).mark_line(
        point=alt.OverlayMarkDef(shape='circle')).encode(x='x',
                                                         y='y',
                                                         color="axis name:",
                                                         tooltip=[alt.Tooltip(title="Variable",field="axis name:"),
                                                                 alt.Tooltip(title="Projection X",field='x'),
                                                                alt.Tooltip(title="Projection Y",field='y')
                                                                 ]
                                                        )
    
    chart = (dash_verti+dash_hori+axes)
    chart.properties(title=f"Cercle des corrélations des composantes {plan_fact[0]+1} et {plan_fact[1]+1}").interactive().display()

def display_proj_indiv(projection, plan_fact, info_cat):

    X_projected = projection[:, plan_fact]
    max_x = max(X_projected[:, 0])
    max_y = max(X_projected[:, 1])


    # affichage des lignes horizontales et verticales
    dash_axis = dict()
    dash_axis["x"] = [0, -1000, 1000, 0, 0]
    dash_axis["y"] = [0, 0, 0, -1000, 1000]
    dash_axis["mode"] = ["a","horizontal", "horizontal", "vertical", "vertical"]

    base_axis = alt.Chart(pd.DataFrame(dash_axis)).mark_line(color='grey').encode(x = alt.X('x', type='quantitative', scale=alt.Scale(domain=[-max_x, max_x]),
                                            axis=alt.Axis(title=f"Composante n°{plan_fact[0]+1}")),
                                            y = alt.Y("y", type='quantitative',scale=alt.Scale(domain=[-max_y, max_y]),
                                            axis=alt.Axis(title=f"Composante n°{plan_fact[1]+1}"))
                                            , strokeDash=alt.StrokeDash("mode", legend=None))

    df =pd.concat([pd.DataFrame(X_projected, columns=["x","y"]),info_cat], axis=1)
    scatter = alt.Chart(df).mark_point().encode(x="x", y="y", color = "Categorie", tooltip=["product_name", 'Categorie'])
    chart = (base_axis+scatter)
    chart.properties(title=f"Projection des individus (sur les composantes {plan_fact[0]+1} et {plan_fact[1]+1})").interactive().display()