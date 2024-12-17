# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:12:32 2023

@author: ythiriet
"""


# Function to create HTML page to ask user data to make prediction
def flight_delay_preparation():


    # Global importation
    from yattag import Doc
    from yattag import indent
    import joblib

    # Data importation
    ARRAY_REPLACEMENT_ALL = joblib.load("./script/data_replacement/array_replacement.joblib")

    # Setting list for prediction
    MONTH_LIST = ["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin",
                  "Juillet", "Aout", "Septembre", "Octobre", "Novembre", "Decembre"]
    MONTH_DAY_LIST = [i for i in range(1,32)]
    DAY_LIST = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    UNIQUE_CARRIERS = ARRAY_REPLACEMENT_ALL[0]    
    ORIGINS = ARRAY_REPLACEMENT_ALL[1]
    DESTINATIONS = ARRAY_REPLACEMENT_ALL[2]

    # Creating HTML
    doc, tag, text, line = Doc(defaults = {'Month': 'Fevrier'}).ttl()

    # Adding pre-head
    doc.asis('<!DOCTYPE html>')
    doc.asis('<html lang="fr">')
    with tag('head'):
        doc.asis('<meta charset="UTF-8">')
        doc.asis('<meta http-equiv="X-UA-Compatible" content = "IE=edge">')
        doc.asis('<link rel="stylesheet" href="./static/style.css">')
        doc.asis('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">')
        doc.asis('<meta name = "viewport" content="width=device-width, initial-scale = 1.0">')

    # Body start
    with tag('body', klass = 'background_brown_light'):
        with tag('form', action = "{{url_for('flight_treatment')}}", method = "POST", enctype = "multipart/form-data"):
            with tag('div', klass = "container"):
                with tag('div', klass = "row"):
                    with tag('div', klass = "col"):
                        line('h1', 'Plane delay prediction', klass = "text-center title purple")
            

                with tag('div', klass = "row justify-content-between"):
                    with tag('div', klass = "col-9"):
                        
                        # Month choice 
                        with tag('div', klass = "row justify-content-around"):
                            with tag('div', klass = "col-md-3"):
                                line('p', 'Please choose the month of your flight', klass = "text-center list_choice")
                            with tag('div', klass = "col"):
                                with tag('div', klass = "row justify-content-around"):
                                    for MONTH in MONTH_LIST:
                                        with tag('div', klass = "col-md-3 radio_text"):
                                            doc.input(name = 'month', type = 'radio', value = MONTH, klass = "radio_1")
                                            text(MONTH)
                                                        
                        # Separation line
                        line('hr','')
                                                      
                        # Day Number
                        with tag('div', klass = "row justify-content-around div2"):
                            with tag('div', klass = "col-md-3"):
                                line('p', 'Please choose the day number', klass = "text-center list_choice")
                            with tag('div', klass = "col-md-8"):
                                with tag('div', klass = "row justify-content-around"):
                                    for MONTH_DAY in MONTH_DAY_LIST:
                                        with tag('div', klass = "col-md-2 radio_text"):
                                            doc.input(name = 'month_day', type = 'radio', value = MONTH_DAY, klass = "radio_2")
                                            text(MONTH_DAY)
                                    
                        # Separation line
                        line('hr','')
    
                    # Earth Image        
                    with tag('div', klass = "col-3"):
                        doc.asis('<img src="/static/globe_terrestre_BACKGROUND.jpg" alt="Globe Terrestre" width=100% height=100% title="Globe Terrestre"/>')
                      
                        
                # Week day
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-9"):
                        with tag('div', klass = "row justify-content-around"):
                            with tag('div', klass = "col-md-3"):
                                line('p', 'Please choose the day', klass = "text-center list_choice")
                            with tag('div', klass = "col-md-8"):
                                with tag('div', klass = "row justify-content-around"):
                                    for DAY in DAY_LIST:
                                        with tag('div', klass = "col-md-3 radio_text"):
                                            doc.input(name = 'day', type = 'radio', value = DAY, klass = "radio_1")
                                            text(DAY)
                                                
                # Separation line
                line('hr','')
                                    
                # Distance to be travelled by plane
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-9"):
                        with tag('div', klass = "row justify-content-between"):
                            with tag('div', klass = "col-md-9"):
                                line('p', 'Please indicate the distance [kilometers]', klass = "text-start list_choice")
                            with tag('div', klass = "col-md-3"):
                                doc.input(name = 'distance', type = 'text', klass = "area_input")
        
                # Separation line
                line('hr','')
                            
                            
                # Carrier
                with tag('div', klass = "row mb-5"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around"):
                            with tag('div', klass = "col-md-3"):
                                line('p', 'Please choose the carrier', klass = "text-center list_choice")
                            with tag('div', klass = "col-md-9"):
                                with tag('div', klass = "row justify-content-around"):
                                    for CARRIER in UNIQUE_CARRIERS:
                                        with tag('div', klass = "col-md-4 radio_text"):
                                            doc.input(name = 'origin', type = 'radio', value = CARRIER[0], klass = "radio_1")
                                            text(CARRIER[0])
                                                
                        # Separation line
                        line('hr','')
                                
                        # Origin
                        with tag('div', klass = "row justify-content-around"):
                            with tag('div', klass = "col-md-3"):
                                line('p', 'Please choose the origin', klass = "text-center list_choice")
                            with tag('div', klass = "col-md-9"):
                                with tag('div', klass = "row justify-content-around"):
                                    for ORIGIN in ORIGINS:
                                        with tag('div', klass = "col-md-2 radio_text"):
                                            doc.input(name = 'destination', type = 'radio', value = ORIGIN[0], klass = "radio_1")
                                            text(ORIGIN[0])
                            
                        # Separation line
                        line('hr','')
                            
                        # Destination
                        with tag('div', klass = "row justify-content-around"):
                            with tag('div', klass = "col-md-3"):
                                line('p', 'Please choose the destination', klass = "text-center list_choice")
                            with tag('div', klass = "col-md-9"):
                                with tag('div', klass = "row justify-content-around"):
                                    for DESTINATION in DESTINATIONS:
                                        with tag('div', klass = "col-md-2 radio_text"):
                                            doc.input(name = 'carrier', type = 'radio', value = DESTINATION[0], klass = "radio_1")
                                            text(DESTINATION[0])
                                                
                    # Rocket Image
                    with tag('div', klass = "col-md"):
                        doc.asis('<img src="/static/Rocket.jpg" alt="Rocket" width=80% height=80% title="Rocket"/>')

                # Submit button
                with tag('div', klass = "row justify-content-center"):
                    with tag('div', klass = "col-6"):
                        with tag('button', id = 'submit_button', name = "action", klass="button", value = 'Predict', onclick="this.classList.toggle('button--loading')"):
                            with tag('span', klass = "button__text"):
                                text("Est-ce que l'avion aura du retard")
                                                        
    # Saving HTML created
    with open(f"./templates/flight_predict.html", "w") as f:
        f.write(indent(doc.getvalue(), indentation = '    ', newline = '\n', indent_text = True))
        f.close()


# Function to make prediction and plotting them for the customer
def flight_delay_prediction(CURRENT_DIRECTORY, MODEL_INPUT):

    # Global importation
    import joblib
    import numpy as np
    from yattag import Doc
    from yattag import indent
    import math
    
    # Global init
    RF_MODEL = False
    NN_MODEL = False
    GB_MODEL = True
    XG_MODEL = False

    # Class creation
    class Data_prediction():
        def __init__(self, ARRAY_REPLACEMENT_ALL, INDEX_REPLACEMENT_ALL, MODEL):
            self.ARRAY_REPLACEMENT_ALL = ARRAY_REPLACEMENT_ALL
            self.INDEX_REPLACEMENT_ALL = INDEX_REPLACEMENT_ALL
            self.MODEL = MODEL
            self.MODEL_INPUT = np.zeros([0])
            self.JS = ""

        
        # Turning word into numbers to make predictions
        def entry_data_modification(self):

            # Init
            MONTH_ARRAY = np.array(["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin",
                          "Juillet", "Aout", "Septembre", "Octobre", "Novembre", "Decembre"])
            DAY_ARRAY = np.array(["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])

            # Replacing month and day
            self.MODEL_INPUT[0] = np.where(self.MODEL_INPUT[0] == MONTH_ARRAY)[0][0]
            self.MODEL_INPUT[1] = int(self.MODEL_INPUT[1])
            self.MODEL_INPUT[2] = np.where(self.MODEL_INPUT[2] == DAY_ARRAY)[0][0]
            self.MODEL_INPUT[3] = int(self.MODEL_INPUT[3])

            # Replacing origin, destination and carrier
            for i, ARRAY_REPLACEMENT in enumerate(self.ARRAY_REPLACEMENT_ALL):
                self.MODEL_INPUT[i + 4] = np.where(self.MODEL_INPUT[i + 4] == ARRAY_REPLACEMENT)[0]
                if self.MODEL_INPUT[i + 4].shape[0] > 0:
                    self.MODEL_INPUT[i + 4] = self.MODEL_INPUT[i + 4][0]
                else:
                    self.MODEL_INPUT[i + 4] = 0

        # Making prediction using model chosen
        def making_prediction(self):
            self.PREDICTION = self.MODEL.predict(self.MODEL_INPUT.reshape(1,-1))
            self.PROBA = self.MODEL.predict_proba(self.MODEL_INPUT.reshape(1,-1))


        # Creating javascript using prediction
        def javascript_result_creation(self):
            
            # Init
            HALFCIRCLE_HEIGHT = 200
            HALFCIRCLE_HEIGHT_POS = int(1.4*HALFCIRCLE_HEIGHT)
            NEEDLE_SIZE = 20
            NEEDLE_ANGLE = 45*math.pi/180
            
            # Creating Canvas to plot graphic
            self.JS += '/* Creation du canvas */\n'
            self.JS += 'const canvas = document.querySelector(".monCanevas");\n'
            self.JS += 'const width = (canvas.width = window.innerWidth);\n'
            self.JS += 'const height = (canvas.height = window.innerWidth*0.5);\n'
            self.JS += 'const ctx = canvas.getContext("2d");\n'
            
            # Function used to convert degree into radian
            self.JS += '\n/* Fonction pour convertir des degrèes en radians */\n'
            self.JS += 'function degToRad(degrees){\n'
            self.JS += '    return (degrees*Math.PI)/180;\n'
            self.JS += '}\n'
            
            # Half-circle construction
            self.JS += "\n/* Creation d'un demi cercle comme base du compteur*/\n"
            self.JS += f'ctx.fillStyle = "rgb({255*math.pow((1-self.PROBA[0][0]),2)},{255*math.pow(self.PROBA[0][0],3)},0)";\n'
            self.JS += 'ctx.beginPath();\n'
            self.JS += f'ctx.arc(window.innerWidth/2,{HALFCIRCLE_HEIGHT_POS},{HALFCIRCLE_HEIGHT},degToRad(180),degToRad(0),false);\n'
            self.JS += 'ctx.fill();\n'
            
            # Building line dash at 50%
            self.JS += "\n/* Creation d'une ligne pointillee au milieu du demi-cercle*/\n"
            self.JS += 'ctx.fillStyle = "rgb(255,255,255)";\n'
            self.JS += 'ctx.beginPath();\n'
            self.JS += 'ctx.setLineDash([5,5]);\n'
            self.JS += f'ctx.moveTo(window.innerWidth/2, {HALFCIRCLE_HEIGHT_POS});\n'
            self.JS += f'ctx.lineTo(window.innerWidth/2, {HALFCIRCLE_HEIGHT_POS - HALFCIRCLE_HEIGHT});\n'
            self.JS += 'ctx.lineWidth = 2;\n'
            self.JS += 'ctx.stroke();\n'
            
            # Building plot needle
            self.JS += "\n/* Creation de l'aguille du compteur */\n"
            self.JS += "ctx.strokeStyle = '#4488EE';\n"
            self.JS += 'ctx.beginPath();\n'
            self.JS += 'ctx.setLineDash([]);\n'
            self.JS += f'ctx.moveTo(window.innerWidth/2, {HALFCIRCLE_HEIGHT_POS});\n'
            self.JS += f'ctx.lineTo(window.innerWidth/2 + {int(HALFCIRCLE_HEIGHT*math.cos(math.pi*(1-self.PROBA[0][0])))}, {HALFCIRCLE_HEIGHT_POS-int(HALFCIRCLE_HEIGHT*math.sin(math.pi*(1-self.PROBA[0][0])))});\n'
            self.JS += 'ctx.lineWidth = 2;\n'
            self.JS += 'ctx.stroke();\n'
            
            self.JS += "ctx.fillStyle = '#4488EE';\n"
            self.JS += 'ctx.beginPath();\n'
            self.JS += f'ctx.moveTo(window.innerWidth/2 + {int(HALFCIRCLE_HEIGHT*math.cos(math.pi*(1-self.PROBA[0][0])))}, {HALFCIRCLE_HEIGHT_POS-int(HALFCIRCLE_HEIGHT*math.sin(math.pi*(1-self.PROBA[0][0])))});\n'
            self.JS += f'ctx.lineTo(window.innerWidth/2 + {int(HALFCIRCLE_HEIGHT*math.cos(math.pi*(1-self.PROBA[0][0]))) - (NEEDLE_SIZE/math.cos(NEEDLE_ANGLE/2))*math.cos(math.pi*(1-self.PROBA[0][0])-NEEDLE_ANGLE/2)}, {HALFCIRCLE_HEIGHT_POS-int(HALFCIRCLE_HEIGHT*math.sin(math.pi*(1-self.PROBA[0][0]))) + (NEEDLE_SIZE/math.cos(NEEDLE_ANGLE/2))*math.sin(math.pi*(1-self.PROBA[0][0])-NEEDLE_ANGLE/2)});\n'
            self.JS += f'ctx.lineTo(window.innerWidth/2 + {int(HALFCIRCLE_HEIGHT*math.cos(math.pi*(1-self.PROBA[0][0]))) - (NEEDLE_SIZE/math.cos(NEEDLE_ANGLE/2))*math.sin(math.pi/2 - math.pi*(1-self.PROBA[0][0]) - NEEDLE_ANGLE/2)}, {HALFCIRCLE_HEIGHT_POS-int(HALFCIRCLE_HEIGHT*math.sin(math.pi*(1-self.PROBA[0][0]))) + (NEEDLE_SIZE/math.cos(NEEDLE_ANGLE/2))*math.cos(math.pi/2 - math.pi*(1-self.PROBA[0][0]) - NEEDLE_ANGLE/2)});\n'
            self.JS += f'ctx.lineTo(window.innerWidth/2 + {int(HALFCIRCLE_HEIGHT*math.cos(math.pi*(1-self.PROBA[0][0])))}, {HALFCIRCLE_HEIGHT_POS-int(HALFCIRCLE_HEIGHT*math.sin(math.pi*(1-self.PROBA[0][0])))});\n'
            self.JS += 'ctx.fill();\n'
            
            # Writing text at both end of plot
            self.JS += "\n/* Creation d'un texte de chaque côté de l'aiguille */\n"
            self.JS += "ctx.fillStyle = 'rgb(0,255,0)';\n"
            self.JS += 'ctx.font = "48px georgia";\n'
            self.JS += f'ctx.fillText("On time",width/2 + 120,{HALFCIRCLE_HEIGHT_POS + 40});\n'
            self.JS += "ctx.fillStyle = 'rgb(255,0,0)';\n"
            self.JS += 'ctx.font = "48px georgia";\n'
            self.JS += f'ctx.fillText("Delayed",width/2 - 300 ,{HALFCIRCLE_HEIGHT_POS + 40});\n'
            
            # Writing text corresponding to the percentage
            self.JS += "\n/* Creation d'un texte au bout de l'aiguille */\n"
            self.JS += f'ctx.fillStyle = "rgb({255*math.pow((1-self.PROBA[0][0]),2)},{255*math.pow(self.PROBA[0][0],3)},0)";\n'
            self.JS += 'ctx.font = "48px georgia"\n'
            self.JS += f'var textString = "{round(100*self.PROBA[0][0],2)} %",\n'
            self.JS += "    textWidth = ctx.measureText(textString).width;\n"
            self.JS += f'ctx.fillText("{round(100*self.PROBA[0][0],2)} %",(width/2) - (textWidth / 2),{HALFCIRCLE_HEIGHT_POS - HALFCIRCLE_HEIGHT - 20})\n'
            
            # Writing Javascript into a file
            with open("./static/main.js","w") as f:
                f.write(self.JS)


        # Creating html
        def html_result_creation(self, CURRENT_DIRECTORY):

            # Creating HTML
            doc, tag, text, line = Doc(defaults = {'Month': 'Fevrier'}).ttl()

            # Adding pre-head
            doc.asis('<!DOCTYPE html>')
            doc.asis('<html lang="fr">')
            with tag('head'):
                doc.asis('<meta charset="UTF-8">')
                doc.asis('<meta http-equiv="X-UA-Compatible" content = "IE=edge">')
                doc.asis('<link rel="stylesheet" href="./static/style.css">')
                doc.asis('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">')
                doc.asis('<meta name = "viewport" content="width=device-width, initial-scale = 1.0">')

            with tag('body', klass = 'background_brown_light'):
                with tag('div', klass = "container"):
                    with tag('div', klass = "row"):
                        with tag('div', klass = "col"):
                            line('h1', 'Plane delay prediction', klass = "text-center title purple")
                        
                        if self.PREDICTION[0]:
                            line('p', f"Selon les donnees, l'avion a {round(100*self.PROBA[0][1],2)} % de chance d'etre retarde", klass = "text-center subtitle")
                        else:
                            line('p', f"Selon les donnees, l'avion a {round(100*self.PROBA[0][0],2)} % de chance d'etre a l'heure", klass = "text-center subtitle")
                    
                    with tag('form', action = "{{url_for('flight_predict')}}", method = "GET", enctype = "multipart/form-data"):
                        with tag('div', klass = "text-center"):
                            with tag('button', id = 'submit_button', name = "action", klass="button", value = 'Go back to previous page', onclick="this.classList.toggle('button--loading')"):
                                with tag('span', klass = "button__text"):
                                    text('Realiser une nouvelle prediction')

                # Javascript
                with tag('canvas', klass = "monCanevas"):
                    line('p', 'Un contenu alternatif ici')

                doc.asis('<script src="/static/main.js"></script>')

            # Saving HTML
            with open("./templates/flight_result.html", "w") as f:
                f.write(indent(doc.getvalue(), indentation = '    ', newline = '\n', indent_text = True))
                f.close()

    # Loading models
    ARRAY_REPLACEMENT_ALL = joblib.load("./script/data_replacement/array_replacement.joblib")
    INDEX_REPLACEMENT_ALL = joblib.load("./script/data_replacement/index_replacement.joblib")
    if RF_MODEL == True:
        with open("./script/models/rf_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif NN_MODEL == True:
        with open("./script/models/nn_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif GB_MODEL == True:
        with open("./script/models/gb_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif XG_MODEL == True:
        with open("./script/models/xg_model.sav", 'rb') as f:
            MODEL = joblib.load(f)

    # Personnalized prediction
    global_data_prediction = Data_prediction(ARRAY_REPLACEMENT_ALL, INDEX_REPLACEMENT_ALL, MODEL)
    global_data_prediction.MODEL_INPUT = MODEL_INPUT
    global_data_prediction.entry_data_modification()

    global_data_prediction.making_prediction()
    global_data_prediction.javascript_result_creation()
    global_data_prediction.html_result_creation(CURRENT_DIRECTORY)
