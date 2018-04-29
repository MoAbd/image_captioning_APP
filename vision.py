
import glob
import pandas as pd
import argparse
import six
import io

from google.cloud import vision
from google.cloud import translate




def translate_text(target, text):
    
	
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text, target_language=target)
    
    
    return u'{}\n'.format(result['translatedText'])
    

def detect_text(path):
    
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    images = glob.glob(path + "/*")
    df = pd.read_csv(path + "/../" + "cnontextimageurl_results.csv",usecols = [0, 1, 2, 3])
    print(df)
    df["embedding"] = len(df) * [""]

    for image_name in images:

        # [START migration_text_detection]
        with io.open(image_name, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        image_url = image_name.split("/")[-1]
        for index, row in df.iterrows():
            if len(texts) > 0:
                
                if type(row["URL"]) is str:
                    url = row["URL"].split("/")[-1]
                    
                    print(url, image_url)
                    if row["URL"].startswith("http") and url == image_url :
                        
                        # Translate
                        if texts[0].locale != 'en':
                            sentence = translate_text("en", texts[0].description.encode('utf-8'))
                            df.loc[index, "embedding"] = sentence
                            print(sentence)
                            break

                        else:
                            sentence = texts[0].description.encode('utf-8')
                            df.loc[index, "embeddimg"] = sentence
                            print(sentence)
                            break
    
    print(df)
    df.to_csv(path + "/../embedding/" + "cnontextimageurl_results.csv")
                        

        
            
       


if __name__ == '__main__':
    
    detect_text("./CLEANED_IMAGES/cnonimages")