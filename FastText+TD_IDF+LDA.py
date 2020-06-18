#!/usr/bin/env python
# coding: utf-8

# In[ ]:


{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "tf-ıdf-fasttext.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "mount_file_id": "1kyrzL_CU9AlFJLQLTrnOTY7qW12khKOK",
      "authorship_tag": "ABX9TyNrXu+cGH7udSJLiejWy6wl",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ozgurugurr/fasttext_tf_-df/blob/master/tf_%C4%B1df_fasttext.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KDxXda-kWiDU",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Google Drive'ı bağlayın ve kodları çalıştırın"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dhM47odskc1T",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%cd drive/My\\ Drive"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "w3Pa1p5Kka0f",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%mkdir tf-ıdf-fasttext\n",
        "%cd tf-ıdf-fasttext"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_hGNgj2oiE0D",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 428
        },
        "outputId": "a3c8024d-7156-4b80-8f50-a2b8e27263fe"
      },
      "source": [
        "import os\n",
        "import spacy\n",
        "import pickle\n",
        "from spacy.lang.en import English\n",
        "from spacy import displacy\n",
        "import numpy as np\n",
        "import tqdm\n",
        "import pandas as pd\n",
        "from gensim.models import FastText\n",
        "from gensim.models.phrases import Phrases, Phraser\n",
        "from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_non_alphanum\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "import xx_ent_wiki_sm\n",
        "\n",
        "## ENGLISH\n",
        "# !python -m spacy download en_core_web_md\n",
        "\n",
        "## MULTILANGUAGE\n",
        "!python -m spacy download xx_ent_wiki_sm\n",
        "nlp = xx_ent_wiki_sm.load()"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: xx_ent_wiki_sm==2.2.0 from https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-2.2.0/xx_ent_wiki_sm-2.2.0.tar.gz#egg=xx_ent_wiki_sm==2.2.0 in /usr/local/lib/python3.6/dist-packages (2.2.0)\n",
            "Requirement already satisfied: spacy>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from xx_ent_wiki_sm==2.2.0) (2.2.4)\n",
            "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (1.0.2)\n",
            "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (3.0.2)\n",
            "Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (1.18.3)\n",
            "Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (7.4.0)\n",
            "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (2.0.3)\n",
            "Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (0.6.0)\n",
            "Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (1.0.0)\n",
            "Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (1.0.2)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (46.1.3)\n",
            "Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (1.1.3)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (2.21.0)\n",
            "Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (0.4.1)\n",
            "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (4.38.0)\n",
            "Requirement already satisfied: importlib-metadata>=0.20; python_version < \"3.8\" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (1.6.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (2020.4.5.1)\n",
            "Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (1.24.3)\n",
            "Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (2.8)\n",
            "Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (3.0.4)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < \"3.8\"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.0->xx_ent_wiki_sm==2.2.0) (3.1.0)\n",
            "\u001b[38;5;2m✔ Download and installation successful\u001b[0m\n",
            "You can now load the model via spacy.load('xx_ent_wiki_sm')\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yxc59JzcIWn5",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Sa_bWapNIWrF",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "pd.options.display.max_colwidth = 800\n",
        "path = \"/content/drive/My Drive/tf-ıdf-fasttext/AllMRLineItems_Mapping_Matched.csv\"\n",
        "combined = pd.read_csv(path, skip_blank_lines= False, sep=',')\n",
        "combined = pd.DataFrame(data = combined)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "b869d4XQEAeu",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!head -n 100 \"/content/drive/My Drive/tf-ıdf-fasttext/AllMRLineItems_Mapping_Matched.csv\" > \"/content/drive/My Drive/tf-ıdf-fasttext/AllMRLineItems_Mapping_Matched_100.csv\""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KMB5mKnLEDKZ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "combined = combined[['MRCreationDate', 'MRID', 'MRTitle', 'MRRemarks', 'MRPrivateNotes',\n",
        "                    'CommodityCode', 'MRType', 'JobNumber', 'Company', 'MRLineItemID',\n",
        "                    'MRLineItemDescription', 'Quantity', 'MaterialType', 'Unit',\n",
        "                    'CostCodeDescription', 'CommonCostCode', 'CommonCostCodeDescription',\n",
        "                    'ProjectID', 'ProjectCode', 'Flag', 'Prefix',\n",
        "                    'CategoryCode', 'CategoryCodeName']]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3oqRmjs17HLe",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "cols = combined.columns.tolist()\n",
        "cols = ['Flag',\n",
        "        'ProjectCode',\n",
        "        'ProjectID',       \n",
        "        'MRID',\n",
        "        'MRLineItemID',\n",
        "        'MRTitle',\n",
        "        'MRLineItemDescription',\n",
        "        'CategoryCode',\n",
        "        'CategoryCodeName',\n",
        "        'CostCodeDescription',\n",
        "        'CommonCostCode',\n",
        "        'CommonCostCodeDescription',\n",
        "        'MRRemarks',\n",
        "        'MRPrivateNotes',\n",
        "        'CommodityCode',\n",
        "        'MRType',\n",
        "        'JobNumber',\n",
        "        'Company',     \n",
        "        'Quantity',\n",
        "        'MaterialType',\n",
        "        'Unit',              \n",
        "        'Prefix'\n",
        "        ]\n",
        "combined = combined[cols]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aeBiEZJI75D3",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 649
        },
        "outputId": "6fcc76e6-cc19-4bc5-9530-ee5c88f43297"
      },
      "source": [
        "combined.head()"
      ],
      "execution_count": 138,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Flag</th>\n",
              "      <th>ProjectCode</th>\n",
              "      <th>ProjectID</th>\n",
              "      <th>MRID</th>\n",
              "      <th>MRLineItemID</th>\n",
              "      <th>MRTitle</th>\n",
              "      <th>MRLineItemDescription</th>\n",
              "      <th>CategoryCode</th>\n",
              "      <th>CategoryCodeName</th>\n",
              "      <th>CostCodeDescription</th>\n",
              "      <th>CommonCostCode</th>\n",
              "      <th>CommonCostCodeDescription</th>\n",
              "      <th>MRRemarks</th>\n",
              "      <th>MRPrivateNotes</th>\n",
              "      <th>CommodityCode</th>\n",
              "      <th>MRType</th>\n",
              "      <th>JobNumber</th>\n",
              "      <th>Company</th>\n",
              "      <th>Quantity</th>\n",
              "      <th>MaterialType</th>\n",
              "      <th>Unit</th>\n",
              "      <th>Prefix</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>2026</td>\n",
              "      <td>90</td>\n",
              "      <td>31163</td>\n",
              "      <td>348608</td>\n",
              "      <td>Roofing Insulation-DOW ROOFMATE(TM) SL-X</td>\n",
              "      <td>DOW ROOFMATE(TM) SL-X Extruded Polystyrene Foam 50X600X1250MM</td>\n",
              "      <td>ATRX</td>\n",
              "      <td>Roofing, Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Hot Rubberized Protected Membrane Roofing Assembly</td>\n",
              "      <td>33ATRX</td>\n",
              "      <td>ATRX - Roofing Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Null</td>\n",
              "      <td>Null</td>\n",
              "      <td>A000</td>\n",
              "      <td>MR</td>\n",
              "      <td>02026-02026</td>\n",
              "      <td>-</td>\n",
              "      <td>17500000.0</td>\n",
              "      <td>Null</td>\n",
              "      <td>m3-Cubic meter</td>\n",
              "      <td>33</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>2026</td>\n",
              "      <td>90</td>\n",
              "      <td>77639</td>\n",
              "      <td>1023998</td>\n",
              "      <td>75216 SBS Modified Bituminous Membrane Roofing - Cap/Base Sheet for UTL</td>\n",
              "      <td>11008 AQUADERE</td>\n",
              "      <td>ATRX</td>\n",
              "      <td>Roofing, Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Hot Rubberized Protected Membrane Roofing Assembly</td>\n",
              "      <td>33ATRX</td>\n",
              "      <td>ATRX - Roofing Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Null</td>\n",
              "      <td>Null</td>\n",
              "      <td>A000</td>\n",
              "      <td>MR</td>\n",
              "      <td>02026-02026</td>\n",
              "      <td>TR001-ENKA İnşaat ve Sanayi A.Ş.</td>\n",
              "      <td>2500000.0</td>\n",
              "      <td>Permanent Material</td>\n",
              "      <td>l-Liter</td>\n",
              "      <td>33</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>2026</td>\n",
              "      <td>90</td>\n",
              "      <td>77639</td>\n",
              "      <td>1023999</td>\n",
              "      <td>75216 SBS Modified Bituminous Membrane Roofing - Cap/Base Sheet for UTL</td>\n",
              "      <td>54766 SOPRALENE S FLAM 180-40 (base / cap sheet)</td>\n",
              "      <td>ATRX</td>\n",
              "      <td>Roofing, Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Hot Rubberized Protected Membrane Roofing Assembly</td>\n",
              "      <td>33ATRX</td>\n",
              "      <td>ATRX - Roofing Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Null</td>\n",
              "      <td>Null</td>\n",
              "      <td>A000</td>\n",
              "      <td>MR</td>\n",
              "      <td>02026-02026</td>\n",
              "      <td>TR001-ENKA İnşaat ve Sanayi A.Ş.</td>\n",
              "      <td>15040000.0</td>\n",
              "      <td>Permanent Material</td>\n",
              "      <td>sqm-Square meter</td>\n",
              "      <td>33</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>2026</td>\n",
              "      <td>90</td>\n",
              "      <td>86556</td>\n",
              "      <td>1148101</td>\n",
              "      <td>75216 - Extra Surface Granule Materials - Soprema</td>\n",
              "      <td>Soprema Paillettes D'ardoise - Gris Chagall</td>\n",
              "      <td>ATRX</td>\n",
              "      <td>Roofing, Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Hot Rubberized Protected Membrane Roofing Assembly</td>\n",
              "      <td>33ATRX</td>\n",
              "      <td>ATRX - Roofing Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Null</td>\n",
              "      <td>Null</td>\n",
              "      <td>A000</td>\n",
              "      <td>MR</td>\n",
              "      <td>02026-02026</td>\n",
              "      <td>TR001-ENKA İnşaat ve Sanayi A.Ş.</td>\n",
              "      <td>1200000.0</td>\n",
              "      <td>Permanent Material</td>\n",
              "      <td>kg-Kilogram</td>\n",
              "      <td>33</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1</td>\n",
              "      <td>2026</td>\n",
              "      <td>90</td>\n",
              "      <td>86556</td>\n",
              "      <td>1148102</td>\n",
              "      <td>75216 - Extra Surface Granule Materials - Soprema</td>\n",
              "      <td>Soprema Paillettes D'ardoise - Ardoise Gris</td>\n",
              "      <td>ATRX</td>\n",
              "      <td>Roofing, Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Hot Rubberized Protected Membrane Roofing Assembly</td>\n",
              "      <td>33ATRX</td>\n",
              "      <td>ATRX - Roofing Self-Adhered Modified Bituminous Membrane</td>\n",
              "      <td>Null</td>\n",
              "      <td>Null</td>\n",
              "      <td>A000</td>\n",
              "      <td>MR</td>\n",
              "      <td>02026-02026</td>\n",
              "      <td>TR001-ENKA İnşaat ve Sanayi A.Ş.</td>\n",
              "      <td>1200000.0</td>\n",
              "      <td>Permanent Material</td>\n",
              "      <td>kg-Kilogram</td>\n",
              "      <td>33</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Flag ProjectCode  ProjectID  ...        MaterialType              Unit Prefix\n",
              "0     1        2026         90  ...                Null    m3-Cubic meter     33\n",
              "1     1        2026         90  ...  Permanent Material           l-Liter     33\n",
              "2     1        2026         90  ...  Permanent Material  sqm-Square meter     33\n",
              "3     1        2026         90  ...  Permanent Material       kg-Kilogram     33\n",
              "4     1        2026         90  ...  Permanent Material       kg-Kilogram     33\n",
              "\n",
              "[5 rows x 22 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 138
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eIlLxgZmEDNM",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# içinde bir kelime listesi bulunan dokümanların bir listesini oluşturur:\n",
        "# örnek : ['wc', 'exhaust', 'fan', 'v', '180m3', 'h', 'h', '100', 'pa']\n",
        "text = []\n",
        "for i in combined.MRLineItemDescription.values:\n",
        "  doc = nlp(remove_stopwords(strip_punctuation(strip_non_alphanum(str(i).lower()))))\n",
        "  tokens = [token.text for token in doc]\n",
        "  text.append(tokens)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ap2t0NC9EDPo",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "common_terms = [\"of\", \"with\", \"without\", \"and\", \"or\", \"the\", \"a\"]\n",
        "# Cümle listesinden ilgili ifadeleri oluşturun:\n",
        "phrases = Phrases(text, common_terms=common_terms, threshold = 10, min_count=5)\n",
        "# Phraser nesnesi cümleleri dönüştürmek için kullanılıyor\n",
        "bigram = Phraser(phrases)\n",
        "tokens = list(bigram[text])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "O5ADWjWlEDT9",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# FastText modelini eğitin\n",
        "model = FastText(tokens, size=100, window=3, min_count=1, iter=10, sorted_vocab=1)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "upQc5b6KAHOl",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!head -n 80000 \"/content/drive/My Drive/tf-ıdf-fasttext/AllMRLineItems_Mapping_Matched.csv\" > \"/content/drive/My Drive/tf-ıdf-fasttext/AllMRLineItems_Mapping_Matched_Test.csv\""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-8_CsxEjD0rA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "I4sa1IqUBBBr",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# def print_results(N, p, r):\n",
        "#     print(\"N\\t\" + str(N))\n",
        "#     print(\"P@{}\\t{:.3f}\".format(1, p))\n",
        "#     print(\"R@{}\\t{:.3f}\".format(1, r))\n",
        "\n",
        "# print_results(*model.test('/content/drive/My Drive/tf-ıdf-fasttext/AllMRLineItems_Mapping_Matched_Test.csv'))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hv25hsxAnM5_",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 241
        },
        "outputId": "33d7e382-fcbe-4bb9-ddd2-9c16f550f70a"
      },
      "source": [
        "model.wv.most_similar('fasteners')"
      ],
      "execution_count": 157,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.\n",
            "  if np.issubdtype(vec.dtype, np.int):\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[('tpfasteners', 0.9931657314300537),\n",
              " ('fastenersthese', 0.980890691280365),\n",
              " ('fastenersst', 0.9805139303207397),\n",
              " ('fasteners_ec', 0.9770998954772949),\n",
              " ('fastenersthis', 0.9761537313461304),\n",
              " ('tabs_fasteners', 0.9714223146438599),\n",
              " ('fasteners_sx5', 0.9680678844451904),\n",
              " ('fasteners_sxc', 0.9680439233779907),\n",
              " ('fasteners_sx14', 0.9606038331985474),\n",
              " ('fastners', 0.9529014825820923)]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 157
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ve8EpmNhnRRp",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 88
        },
        "outputId": "27ac1733-90fb-4c30-9ca6-c5a39a3c9c88"
      },
      "source": [
        "model.wv.similarity('fastener', 'fast')"
      ],
      "execution_count": 149,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.\n",
            "  if np.issubdtype(vec.dtype, np.int):\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7136766"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 149
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qBh-H3xAoauo",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 241
        },
        "outputId": "2176df73-f888-4266-9a9a-6a8799c53a38"
      },
      "source": [
        "model.wv.most_similar('polystyrene')      "
      ],
      "execution_count": 146,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.\n",
            "  if np.issubdtype(vec.dtype, np.int):\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[('polystyrene_foam', 0.9640012383460999),\n",
              " ('polythene', 0.9502295255661011),\n",
              " ('polypropylene', 0.9262514710426331),\n",
              " ('polythelene', 0.9242044687271118),\n",
              " ('polysil', 0.9225318431854248),\n",
              " ('polypol', 0.9212021827697754),\n",
              " ('polyproplene', 0.9208230972290039),\n",
              " ('polysyrlyn', 0.9153357148170471),\n",
              " ('polypropilen', 0.914897620677948),\n",
              " ('polyfoam', 0.9125882387161255)]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 146
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3HdUNJHLofTa",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 88
        },
        "outputId": "ecf07083-83a5-4a37-d5c5-89e4dac4fc7a"
      },
      "source": [
        "model.wv.similarity('polystyrene', 'polystyren')"
      ],
      "execution_count": 150,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.\n",
            "  if np.issubdtype(vec.dtype, np.int):\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9777656"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 150
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4j9yLAsEqskv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# # kelime vektör modeli kaydet\n",
        "# model.save_model(\"/content/drive/My Drive/tf-ıdf-fasttext/model.bin\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YOtF0LSJqsnc",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# # kelime vektör modeli yükle\n",
        "# model = fasttext.load_model(\"/content/drive/My Drive/tf-ıdf-fasttext/model.bin\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RGxTuB9tEDSn",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "325f9809-7267-4f44-c70d-d31a9dfdab15"
      },
      "source": [
        "# TF-IDF fasttext 'model' ile birlikte kelimeler ve dokümanlar için bir liste listesine ihtiyaç duyar.\n",
        "from tqdm import tqdm\n",
        "\n",
        "text = []\n",
        "for i in tqdm(tokens):\n",
        "  string = ' '.join(i)\n",
        "  text.append(string)\n",
        "tf_idf_vect = TfidfVectorizer(stop_words=None)\n",
        "final_tf_idf = tf_idf_vect.fit_transform(text)\n",
        "tfidf_feat = tf_idf_vect.get_feature_names()"
      ],
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "100%|██████████| 434598/434598 [00:00<00:00, 1877068.39it/s]\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dp_zeozYEL3c",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 88
        },
        "outputId": "969ba859-3d74-406f-fae2-432260f03ea8"
      },
      "source": [
        "# np.seterr(divide='ignore', invalid='ignore') ??\n",
        "tfidf_sent_vectors = []; \n",
        "row=0;\n",
        "errors=0\n",
        "for sent in tqdm(tokens): \n",
        "    sent_vec = np.zeros(100) \n",
        "    weight_sum =0; \n",
        "    for word in sent: \n",
        "        try:\n",
        "            vec = model.wv[word]\n",
        "            # cümle / inceleme içindeki bir kelimenin tf_idfidf değerini elde etmek\n",
        "            tfidf = final_tf_idf [row, tfidf_feat.index(word)]\n",
        "            sent_vec += (vec * tfidf)\n",
        "            weight_sum += tfidf\n",
        "        except:\n",
        "            errors =+1\n",
        "            pass\n",
        "    sent_vec /= weight_sum\n",
        "    #print(np.isnan(np.sum(sent_vec))) ??\n",
        "\n",
        "    tfidf_sent_vectors.append(sent_vec)\n",
        "    row += 1\n",
        "print('errors noted: '+str(errors))"
      ],
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "  1%|          | 2457/434598 [00:31<1:35:27, 75.45it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:17: RuntimeWarning: invalid value encountered in true_divide\n",
            "100%|██████████| 434598/434598 [1:27:12<00:00, 83.05it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "errors noted: 1\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2ynH0Xt25fpy",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "with open(\"/content/drive/My Drive/tf-ıdf-fasttext/tfidf_sent_vectors.pkl\", 'wb') as handle:\n",
        "                    pickle.dump(tfidf_sent_vectors, handle)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fGDJcYTmEL6l",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# bu vektörleri tekrar veri çerçevesine birleştir:\n",
        "combined['FT_tfidf'] = tfidf_sent_vectors"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vuI9zmehEL81",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 207
        },
        "outputId": "f16110d6-7c94-4d8f-944c-c1dcc1f5de7a"
      },
      "source": [
        "keyword = 'fasteners'\n",
        "\n",
        "query = [combined.loc[combined.MRLineItemDescription.str.contains(keyword)].iloc[0]['FT_tfidf']]\n",
        "query = np.array(list(query))\n",
        "query = np.nan_to_num(query)\n",
        "\n",
        "vectors = np.array(list(combined.FT_tfidf.values))\n",
        "vectors = np.nan_to_num(vectors)\n",
        "\n",
        "cosine_similarities = pd.Series(cosine_similarity(query, vectors).flatten())\n",
        "\n",
        "for i,j in cosine_similarities.nlargest(10).iteritems():\n",
        "  print(str(i) + '-' + combined.MRLineItemDescription.iloc[i] + \" \" + str(j))"
      ],
      "execution_count": 107,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "3108-Metal Roof Panels (Gutter Downspouts with all fasteners) spec 74113 1.0\n",
            "3109-Metal Wall Panel (CUSTOM STAINLESS STEEL CLADDING SYSTEM with all accessories and fasteners) spec 74213 0.9053069646681828\n",
            "68056-Roof Screen Wall Panels Complete to include clips fasteners and any intermediate support required to maintain panel integrity. 0.8998648038409534\n",
            "24940-Linear Wood Ceilings and Panels (WP3) - Aluratone 750 Microperf Walls Ceilings Soffits & Fascia's.  3/4 Thick x up to 4' Wide x upto 8' or 10' Long.  Species: Harward Veneer as per specifications QS/BM.  Finish Custom Stain to March Architect's Sample.  ASTME-8 Fire Rated.  Includes Z Clips Torsion Springs and Saddle Climps and heavy uty Metric Grid.   0.8959730978227042\n",
            "24941-Panel Grille Ceilings (WPC3) - Panel Grille Ceilings as per Drawings - Wood Backed.  Species: Rulon Harwood as per specifications.  Finish: Custom Stain to Match Architect's Sample.  ASTME-84 Fire Rated.  Includes Black Fabric Backer Wood Backer Clips Heavy Duty Metric Grid Unsulation. 0.8946931441516595\n",
            "3107-Metal Roof Panels (Costum SS Roofing System) with all components waterproofing membrane xps insulation alu battens connection members spec 74113 0.8930038641776756\n",
            "341556-074113.16STANDING-SEAM METAL ROOF PANELS 0.8855886851029335\n",
            "32507-FWP-4 : DECOUSTIC ACOUSTIC WALL PANEL HIGH IMPACT RESISTANT TYPE 1 (HIR1) PANELS (TACKABLE) AS PER ASTM E84_MOUNTING INCLUDED_FINISH:FR-701 Silver Papier (SP) 0.8845092170657014\n",
            "169138-SPX Penthouse Acoustical Metal Panels 0.8815036425774221\n",
            "327230-LINK BRIDGE Standin Seam Roofing System For Precurved Roof Material 0.879910506146595\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kxkKFGVY2BIK",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 54
        },
        "outputId": "0d79cf2e-b1fb-4fc0-9b73-c522785b3037"
      },
      "source": [
        "print(combined.loc[combined.MRLineItemDescription.str.contains(keyword)]['MRLineItemDescription'].values[0])"
      ],
      "execution_count": 113,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Metal Wall Panel (CUSTOM STAINLESS STEEL CLADDING SYSTEM with all accessories and fasteners) spec 74213\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "L5Dzmoct3YxF",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "keyword = 'A000'\n",
        "\n",
        "query = [combined.loc[combined.CategoryCode.str.contains(keyword)].iloc[0]['FT_tfidf']]\n",
        "query = np.array(list(query))\n",
        "query = np.nan_to_num(query)\n",
        "\n",
        "vectors = np.array(list(combined.FT_tfidf.values))\n",
        "vectors = np.nan_to_num(vectors)\n",
        "\n",
        "cosine_similarities = pd.Series(cosine_similarity(query, vectors).flatten())\n",
        "\n",
        "for i,j in cosine_similarities.nlargest(10).iteritems():\n",
        "  print(str(i) + '-' + combined.CategoryCode.iloc[i] + \" \" + str(j))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4y70Ayvf3Y5i",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "print(combined.loc[combined.MRLineItemDescription.str.contains(keyword)]['MRLineItemDescription'].values[0])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kWF_Wp8o0ldP",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 187
        },
        "outputId": "0438e297-7d7d-41cb-8f0d-8b0c11430254"
      },
      "source": [
        "keyword = 'ATWP'\n",
        "\n",
        "query = [combined.loc[combined.CommonCostCode.str.contains(keyword)].iloc[0]['FT_tfidf']]\n",
        "query = np.array(list(query))\n",
        "query = np.nan_to_num(query)\n",
        "\n",
        "vectors = np.array(list(combined.FT_tfidf.values))\n",
        "vectors = np.nan_to_num(vectors)\n",
        "\n",
        "cosine_similarities = pd.Series(cosine_similarity(query, vectors).flatten())\n",
        "\n",
        "for i,j in cosine_similarities.nlargest(10).iteritems():\n",
        "  print(str(i) + '-' + combined.CommonCostCode.iloc[i] + \" \" + str(j))"
      ],
      "execution_count": 117,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "26322-33ATWP 0.9999999999999998\n",
            "26323-33ATWP 0.9999999999999998\n",
            "398626-33AFFN 0.923457699201812\n",
            "29333-33ATJS 0.9194347571755452\n",
            "29434-33ATJS 0.9194347571755452\n",
            "24589-33AMSC 0.9160184619556551\n",
            "24747-33AMSC 0.9160184619556551\n",
            "398618-33AFFN 0.9160102778487015\n",
            "398627-33AFFN 0.9160102778487015\n",
            "398635-33AFFN 0.9160102778487015\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eDbci3gXEMBc",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "a4bfc890-dc40-413b-fa61-fdd205e1387a"
      },
      "source": [
        "print(combined.loc[combined.CommonCostCode.str.contains(keyword)]['CommonCostCode'].values[0])"
      ],
      "execution_count": 118,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "33ATWP\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QZAQ4M-hEMDi",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# combined = combined['FT_tfidf']\n",
        "# combined.to_csv( \"/content/drive/My Drive/tf-ıdf-fasttext/FT_tfidf.csv\", index=False) "
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}

