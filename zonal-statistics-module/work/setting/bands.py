import ee

def maskModisQA(image):
    qa = image.select('SummaryQA')
    MODLAND_QA_Bits = 1<<1
    mask = qa.bitwiseAnd(MODLAND_QA_Bits).eq(0)
    return image.updateMask(mask).divide(10000)

def lst_filter(image):
    
    qaday = image.select(['QC_Day']); 
    qanight = image.select(['QC_Night'])
    dayshift = qaday.rightShift(6)
    nightshift = qanight.rightShift(6)
    daymask = dayshift.lte(2)
    nightmask = nightshift.lte(2)
    #dayshift1 = qaday.leftShift(6)
    #dayshift2 = qaday.rightShift(6)
    #daymask = dayshift2.eq(0)
    #nightshift1 = qanight.leftShift(6)
    #nightshift2 = qanight.rightShift(6)
    #nightmask = nightshift2.eq(0)
    outimage = ee.Image(image.select(['LST_Day_1km', 'LST_Night_1km']))
    outmask = ee.Image([daymask, nightmask])
    return outimage.updateMask(outmask)

def lst_day(image):
    
    lst_day = image.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename("LST_Day")
    image = image.addBands(lst_day)

    return(image)

def lst_night(image):
    
    lst_night = image.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename("LST_Night")
    image = image.addBands(lst_night)

    return(image)

def lst_mean(image):
    lst_mean = image.expression(
    '(day + night) / 2', {
    'day': image.select('LST_Day'),
    'night': image.select('LST_Night')}).rename('LST_Mean')

    return image.addBands(lst_mean)