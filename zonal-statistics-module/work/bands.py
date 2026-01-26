import math

def RH(image):
    
    RH = 100 * math.exp(17.625 * image.select('dewpoint_temperature_2m') / (243.04 + image.select('dewpoint_temperature_2m'))) / math.exp(17.625 * image.select('temperature_2m') / (243.04 + image.select("temperature_2m"))).rename("Relative_Humidity")
    image = image.addBands(RH)

    return(image)