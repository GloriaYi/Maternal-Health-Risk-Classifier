FROM condaforge/miniforge3:25.11.0-0

# copy the lockfile into the container
COPY conda-lock.yml conda-lock.yml

# setup conda-lock and install packages from lockfile
RUN conda install -n base -c conda-forge conda-lock -y
RUN conda-lock install -n dockerlock conda-lock.yml
RUN conda run -n dockerlock pip install deepchecks==0.19.1

# expose JupyterLab port
EXPOSE 8888

# sets the default working directory
# this is also specified in the compose file
# adding test comment to trigger rebuild
WORKDIR /workplace

# run JupyterLab on container start
# uses the jupyterlab from the install environment
CMD ["conda", "run", "--no-capture-output", "-n", "dockerlock", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]