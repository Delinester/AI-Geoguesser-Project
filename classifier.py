import tensorflow as tf
import numpy as np
import os


COUNTRIES = {'Australia':["melbourne"],
 'Brazil':["saopaulo"],
 'Canada':['ottawa', 'toronto'],
 'Denmark':["cph"],
 'Finland':["helsinki"],
 'France':["paris", "strasbourg"],
 'Germany':None,
 'Hungary':["budapest"],
 'India':["bengaluru", "goa"],
 'Japan':["tokyo"],
 'Netherlands':["amsterdam"],
 'Russia':["Moscow"],
 'Sweden':["stockholm"],
 'Thailand':["bangkok"],
 'UnitedStates': ['Austin', 'Boston', 'Miami', 'Phoenix', 'Sf']}


countries_classifier = tf.keras.models.load_model("Countries_classifier.keras")
US_cities_classifer = tf.keras.models.load_model("US_classifier.keras")
Canada_cities_classifier = tf.keras.models.load_model("Canada_classifier.keras") 


def get_top_countries(img_array, top_k=3):
    logits = countries_classifier.predict(img_array, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    # Get indices of top K probabilities
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        country_name = list(COUNTRIES.keys())[idx]
        confidence = probabilities[idx] * 100
        results.append((country_name, confidence))
    
    return results


def classifyCity(img_array, country):
    model = None
    if country == 'UnitedStates':
        model = US_cities_classifer
    elif country == 'Canada':
        model = Canada_cities_classifier
    else:
        try:
            result = ""
            for city in list(COUNTRIES[country]):
                result += city + " OR "
            return result[:-4], 0.0
        except Exception as e:
            return "Undefined", 0.0
    
    logits = model.predict(img_array, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1)
    pred_class = np.argmax(probabilities, axis=1)[0]
    confidence = np.max(probabilities) * 100
    return list(COUNTRIES[country])[pred_class], confidence


def classifyImage(img_path):
    target_size = (256, 256) 
    img = tf.keras.preprocessing.image.load_img(img_path) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)  
    img_array = tf.image.resize_with_pad(
        img_array,
        target_height=target_size[0],
        target_width=target_size[1]
    )
    img_array = tf.expand_dims(img_array, axis=0)
    
    top_countries = get_top_countries(img_array, top_k=3)
    # Use the top prediction for city classification
    top_country, top_conf = top_countries[0]
    city, city_conf = classifyCity(img_array, top_country)
    
    return top_country, top_conf, city, city_conf, top_countries
