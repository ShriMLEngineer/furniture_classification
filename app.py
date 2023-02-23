{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ShriMLEngineer/furniture_classification/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install fastapi uvicorn torch torchvision Pillow python-multipart"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GPZ1o-auZC5M",
        "outputId": "8967c18b-21d7-41df-b00a-dc5bc934f758"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting fastapi\n",
            "  Downloading fastapi-0.92.0-py3-none-any.whl (56 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.2/56.2 KB\u001b[0m \u001b[31m1.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting uvicorn\n",
            "  Downloading uvicorn-0.20.0-py3-none-any.whl (56 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.9/56.9 KB\u001b[0m \u001b[31m7.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: torch in /usr/local/lib/python3.8/dist-packages (1.13.1+cu116)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.8/dist-packages (0.14.1+cu116)\n",
            "Requirement already satisfied: Pillow in /usr/local/lib/python3.8/dist-packages (7.1.2)\n",
            "Collecting python-multipart\n",
            "  Downloading python-multipart-0.0.5.tar.gz (32 kB)\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2 in /usr/local/lib/python3.8/dist-packages (from fastapi) (1.10.5)\n",
            "Collecting starlette<0.26.0,>=0.25.0\n",
            "  Downloading starlette-0.25.0-py3-none-any.whl (66 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m66.4/66.4 KB\u001b[0m \u001b[31m6.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: click>=7.0 in /usr/local/lib/python3.8/dist-packages (from uvicorn) (7.1.2)\n",
            "Collecting h11>=0.8\n",
            "  Downloading h11-0.14.0-py3-none-any.whl (58 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m58.3/58.3 KB\u001b[0m \u001b[31m6.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: typing-extensions in /usr/local/lib/python3.8/dist-packages (from torch) (4.5.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from torchvision) (1.22.4)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.8/dist-packages (from torchvision) (2.25.1)\n",
            "Requirement already satisfied: six>=1.4.0 in /usr/local/lib/python3.8/dist-packages (from python-multipart) (1.15.0)\n",
            "Collecting anyio<5,>=3.4.0\n",
            "  Downloading anyio-3.6.2-py3-none-any.whl (80 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m80.6/80.6 KB\u001b[0m \u001b[31m6.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: chardet<5,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from requests->torchvision) (4.0.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests->torchvision) (2022.12.7)\n",
            "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests->torchvision) (1.24.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests->torchvision) (2.10)\n",
            "Collecting sniffio>=1.1\n",
            "  Downloading sniffio-1.3.0-py3-none-any.whl (10 kB)\n",
            "Building wheels for collected packages: python-multipart\n",
            "  Building wheel for python-multipart (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for python-multipart: filename=python_multipart-0.0.5-py3-none-any.whl size=31678 sha256=574fec127e362fe91d574cf0f997f5343e86a67ffb8e73b80c76087e7f036642\n",
            "  Stored in directory: /root/.cache/pip/wheels/9e/fc/1c/cf980e6413d3ee8e70cd8f39e2366b0f487e3e221aeb452eb0\n",
            "Successfully built python-multipart\n",
            "Installing collected packages: sniffio, python-multipart, h11, uvicorn, anyio, starlette, fastapi\n",
            "Successfully installed anyio-3.6.2 fastapi-0.92.0 h11-0.14.0 python-multipart-0.0.5 sniffio-1.3.0 starlette-0.25.0 uvicorn-0.20.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from fastapi import FastAPI, File, UploadFile\n",
        "from PIL import Image\n",
        "import io\n",
        "import torch\n",
        "import torchvision.transforms as T\n",
        "import asyncio\n",
        "\n",
        "app = FastAPI()\n",
        "\n",
        "path = '/content/furniture_classification/detr-main'\n",
        "\n",
        "model = torch.hub.load(path, 'detr_resnet50', source=\"local\", pretrained=True, num_classes=4)\n",
        "checkpoint = torch.load(\"/content/furniture_classification/outputs/checkpoint.pth\", map_location='cpu')\n",
        "\n",
        "model.load_state_dict(checkpoint['model'], strict=False)\n",
        "\n",
        "@app.post('/predict')\n",
        "async def predict(file: UploadFile = File(...)):\n",
        "\n",
        "  contents = await file.read()\n",
        "  img = Image.open(io.BytesIO(contents)).convert('RGB')\n",
        "  img = img.resize((224,224))\n",
        "  img = torch.unsqueeze(T.ToTensor()(img), 0)\n",
        "  pred = model(img)\n",
        "\n",
        "  predicted_class = torch.argmax(pred).item()\n",
        "\n",
        "  return{'class': predicted_class}"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 293
        },
        "id": "tt2hJjunY-08",
        "outputId": "0e637f72-3190-4398-fd95-8a0175d7a7db"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "FileNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-3-6f54e457cb67>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0mpath\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'/content/furniture_classification/detr-main'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhub\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'detr_resnet50'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msource\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"local\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpretrained\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_classes\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m \u001b[0mcheckpoint\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"/content/furniture_classification/outputs/checkpoint.pth\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmap_location\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'cpu'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/torch/hub.py\u001b[0m in \u001b[0;36mload\u001b[0;34m(repo_or_dir, model, source, trust_repo, force_reload, verbose, skip_validation, *args, **kwargs)\u001b[0m\n\u001b[1;32m    540\u001b[0m                                            verbose=verbose, skip_validation=skip_validation)\n\u001b[1;32m    541\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 542\u001b[0;31m     \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_load_local\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrepo_or_dir\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    543\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    544\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/torch/hub.py\u001b[0m in \u001b[0;36m_load_local\u001b[0;34m(hubconf_dir, model, *args, **kwargs)\u001b[0m\n\u001b[1;32m    567\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    568\u001b[0m     \u001b[0mhubconf_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhubconf_dir\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mMODULE_HUBCONF\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 569\u001b[0;31m     \u001b[0mhub_module\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_import_module\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mMODULE_HUBCONF\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhubconf_path\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    570\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    571\u001b[0m     \u001b[0mentry\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_load_entry_from_hubconf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhub_module\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/torch/hub.py\u001b[0m in \u001b[0;36m_import_module\u001b[0;34m(name, path)\u001b[0m\n\u001b[1;32m     88\u001b[0m     \u001b[0mmodule\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mimportlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mutil\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodule_from_spec\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mspec\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     89\u001b[0m     \u001b[0;32massert\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mspec\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloader\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mLoader\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 90\u001b[0;31m     \u001b[0mspec\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloader\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexec_module\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodule\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     91\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mmodule\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     92\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.8/importlib/_bootstrap_external.py\u001b[0m in \u001b[0;36mexec_module\u001b[0;34m(self, module)\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.8/importlib/_bootstrap_external.py\u001b[0m in \u001b[0;36mget_code\u001b[0;34m(self, fullname)\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.8/importlib/_bootstrap_external.py\u001b[0m in \u001b[0;36mget_data\u001b[0;34m(self, path)\u001b[0m\n",
            "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '/content/furniture_classification/detr-main/hubconf.py'"
          ]
        }
      ]
    }
  ]
}