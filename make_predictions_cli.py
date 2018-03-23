import sys
from sklearn.externals import joblib

# Load the model we trained previously
model = joblib.load('/home/ubuntu/temp/ml/trained_house_classifier_model.pkl')

# For the house we want to value, we need to provide the features in the exact same
# arrangement as our training data set.
house_to_value = [
    # House features
    int(sys.argv[1]),   # year_built
    int(sys.argv[2]),      # stories
    int(sys.argv[3]),      # num_bedrooms
    int(sys.argv[4]),      # full_bathrooms
    int(sys.argv[5]),      # half_bathrooms
    int(sys.argv[6]),   # livable_sqft
    int(sys.argv[7]),   # total_sqft
    int(sys.argv[8]),      # garage_sqft
    int(sys.argv[9]),      # carport_sqft
    sys.argv[10] == '1',   # has_fireplace
    sys.argv[11] == '1',  # has_pool
    sys.argv[12] == '1',   # has_central_heating
    sys.argv[13] == '1',   # has_central_cooling

    # Garage type: Choose only one
    int(sys.argv[14]),      # attached
    int(sys.argv[15]),      # detached
    int(sys.argv[16]),      # none

    # City: Choose only one
    int(sys.argv[17]),      # Amystad
    int(sys.argv[18]),      # Brownport
    int(sys.argv[19]),      # Chadstad
    int(sys.argv[20]),      # Clarkberg
    int(sys.argv[21]),      # Coletown
    int(sys.argv[22]),      # Davidfort
    int(sys.argv[23]),      # Davidtown
    int(sys.argv[24]),      # East Amychester
    int(sys.argv[25]),      # East Janiceville
    int(sys.argv[26]),      # East Justin
    int(sys.argv[27]),      # East Lucas
    int(sys.argv[28]),      # Fosterberg
    int(sys.argv[29]),      # Hallfort
    int(sys.argv[30]),      # Jeffreyhaven
    int(sys.argv[31]),      # Jenniferberg
    int(sys.argv[32]),      # Joshuafurt
    int(sys.argv[33]),      # Julieberg
    int(sys.argv[34]),      # Justinport
    int(sys.argv[35]),      # Lake Carolyn
    int(sys.argv[36]),      # Lake Christinaport
    int(sys.argv[37]),      # Lake Dariusborough
    int(sys.argv[38]),      # Lake Jack
    int(sys.argv[39]),      # Lake Jennifer
    int(sys.argv[40]),      # Leahview
    int(sys.argv[41]),      # Lewishaven
    int(sys.argv[42]),      # Martinezfort
    int(sys.argv[43]),      # Morrisport
    int(sys.argv[44]),      # New Michele
    int(sys.argv[45]),      # New Robinton
    int(sys.argv[46]),      # North Erinville
    int(sys.argv[47]),      # Port Adamtown
    int(sys.argv[48]),      # Port Andrealand
    int(sys.argv[49]),      # Port Daniel
    int(sys.argv[50]),      # Port Jonathanborough
    int(sys.argv[51]),      # Richardport
    int(sys.argv[52]),      # Rickytown
    int(sys.argv[53]),      # Scottberg
    int(sys.argv[54]),      # South Anthony
    int(sys.argv[55]),      # South Stevenfurt
    int(sys.argv[56]),      # Toddshire
    int(sys.argv[57]),      # Wendybury
    int(sys.argv[58]),      # West Ann
    int(sys.argv[59]),      # West Brittanyview
    int(sys.argv[60]),      # West Gerald
    int(sys.argv[61]),      # West Gregoryview
    int(sys.argv[62]),      # West Lydia
    int(sys.argv[63])       # West Terrence
]

# scikit-learn assumes you want to predict the values for lots of houses at once, so it expects an array.
# We just want to look at a single house, so it will be the only item in our array.
homes_to_value = [
    house_to_value
]

# Run the model and make a prediction for each house in the homes_to_value array
predicted_home_values = model.predict(homes_to_value)

# Since we are only predicting the price of one house, just look at the first prediction returned
predicted_value = predicted_home_values[0]

print("{:,.2f}".format(predicted_value))
