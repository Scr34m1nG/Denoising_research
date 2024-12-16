# How to run this project

## Setup venv

> python -m venv .venv
>
> .venv\Scripts\Activate.ps1

## Install the library

install the library from **requirement.txt**. </br>
> pip install -r requirement.txt </br>

## Run the Streamlit

you can run the streamlit using this command.
> streamlit run Denoising_web.py </br>

</br>
then the browser will appear and the website is ready to be used for denoising.</br></br>

## parameters:</br>

Deconvolutions:

- kernel_size: it means the kernel size.
- sigma: the larger this value, the more blurry the image will be.
- K: it means the denoise power.
</br>

Non-Local Means:

- patch_size: the size of the patch to be denoised.
- search_window: size to find a value that is close to the pixel value.
- h: it means the denoise power.

Tensor Decomposition:

- rank: how much the image will be split and will be denoised

## after using it you can deactivate

> deactivate

NOTE: you can also see the results of the experiment in excel file.
