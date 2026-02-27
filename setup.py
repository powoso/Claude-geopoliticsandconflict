from setuptools import setup, find_packages

setup(
    name="geopolitical-prediction-markets",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pgmpy>=0.0.23",
        "networkx>=3.2.0",
        "textblob>=0.17.0",
        "sqlalchemy>=2.0.0",
        "streamlit>=1.29.0",
        "plotly>=5.18.0",
        "folium>=0.15.0",
        "geopandas>=0.14.0",
        "schedule>=1.2.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "feedparser>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "geopred=geopolitical_prediction.cli:main",
        ],
    },
)
