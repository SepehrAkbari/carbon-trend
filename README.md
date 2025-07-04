# CO2 Emissions Trend Analysis

This project utilizes gradient descent to analyze CO2 emissions trends over time, aiming to optimize various models for better prediction accuracy. The goal of the project is to explore different models and perform hyper-parameter tuning and error minimization for each.

## Data

The data used is a time series dataset of CO2 emissions, reported yearly from 1940 to 2024. The dataset is available in the [data](/data/) directory.

## Usage

To run the models, first clone this repository, and run each of the models using their respective scripts:

```bash
git clone https://github.com/SepehrAkbari/carbon-trend.git
cd carbon-trend/src
 
python model_linear.py
python model_sinusoidal.py
python model_exponential.py
python model_sqrt.py

cd ../
```

You can also review our analysis in the [notebook](/notebook/carbon-trend.ipynb) directory:

```bash
cd notebook
jupyter notebook carbon-trend.ipynb
```

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](LICENSE).