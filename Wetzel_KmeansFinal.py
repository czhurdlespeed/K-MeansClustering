import marimo

__generated_with = "0.3.6-dev6"
app = marimo.App()


@app.cell
def __():
    import micropip
    return micropip,


@app.cell
async def __(micropip):
    import marimo as mo
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.datasets import make_blobs
    await micropip.install("altair")
    import altair as alt
    import pandas as pd
    import timeit
    import urllib.request
    import requests
    import PIL.Image
    from pathlib import Path
    from matplotlib import pyplot as plt
    from io import BytesIO
    return (
        BytesIO,
        KMeans,
        PIL,
        Path,
        alt,
        make_blobs,
        mo,
        np,
        pd,
        plt,
        requests,
        silhouette_score,
        timeit,
        urllib,
    )


@app.cell
def __(mo):
    mo.md("""
    <h1 align="center", style="font-weight: bold;">COSC594 Algorithms Presentation</h1>
    <h2 align="center", style="font-weight: bold;">K-Means Clustering</h2>
    <h3 align="center">Calvin Wetzel</h3>
    <h4 align="center">April 2, 2024</h4>
    """)
    return


@app.cell
def __(mo):
    mo.center(mo.md("""##**Sections**"""))
    return


@app.cell
def __(mo):
    mo.md("""
    #####**1. Mathematical Foundations**
    #####**2. Initializing Dataset with Clusters**
    #####**3. Cluster Centroid Initialization**
    #####**4. Cluster Updates**
    #####**5. Model Performance Measures**
    #####**6. Big O Runtimes**
    #####**7. Image Segmentation Example**
    """)
    return


@app.cell
def __(mo):
    mo.center(mo.md("##Mathematical Foundations"))
    return


@app.cell
def __(mo):
    mo.md(
       """
       <p><ul> 
           <li>Randomly Place Initial Centroids</li>
           <ul>
               <li>Calculate Euclidean Distance between each cluster and all data points</li>
               <li>Assign each data point to cluster that results in smallest Euclidean Distance</li>
               <li>Update Cluster Centroid based on newly assigned points</li>
           </ul>
           </ul>
       """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""
    Data set contains points $x_1, x_2,...,x_p.\\$
    Each cluster contains a set of points $S_k.\\$
    Cluster's Points: $S_k = \{$ p $\vert$ if $x_p$ belongs to the $k^{th}$ $cluster\}\\$
    Centroid:
    $c_k = \frac{1}{\mid S_k \mid} \sum_{p\in S_k} x_p\\$
    Point Assigment to Cluster: 
    $a_p = argmin_{k_1,...,K} \lVert x_p - c_k \rVert_2$
    """)
    return


@app.cell
def __(mo):
    mo.center(mo.md("##Initializing Dataset with Clusters"))
    return


@app.cell
def __(mo):
    # Make sliders for stds and number of centers
    num_clusters = mo.ui.slider(2, 25, step=1, label="Number of Clusters: ")
    num_clusters
    return num_clusters,


@app.cell
def __(np, num_clusters):
    # Cluster standard deviation
    stds = np.squeeze(np.random.exponential(3, size=(1, num_clusters.value)))
    return stds,


@app.cell
def __(mo):
    datapoints = mo.ui.slider(100, 7500, step=100, value=2400, label='Number of Datapoints: ')
    datapoints
    return datapoints,


@app.cell
def __(alt, datapoints, make_blobs, mo, num_clusters, pd, stds):
    X, y = make_blobs(n_samples=datapoints.value, centers=num_clusters.value, cluster_std=stds, random_state=0, center_box=(-25,25))
    # Turn into Pands dataframe
    X_df = pd.DataFrame(X, columns=["x1", "x2"])
    size_slider = alt.binding_range(min=1, max=20, step=1, name="Point Size: ")
    size_var = alt.param(bind=size_slider, value=10)
    chart = alt.Chart(X_df).mark_point(
        filled=True,
        size=size_var,
    ).encode(x="x1:Q",y="x2:Q").add_params(
        size_var
    )
    chart = mo.ui.altair_chart(chart)
    return X, X_df, chart, size_slider, size_var, y


@app.cell
def __(chart, mo):
    mo.vstack([chart, chart.value.head()])
    return


@app.cell
def __(mo):
    mo.center(mo.md("##Cluster Centroid Initialization"))
    return


@app.cell
def __(KMeans, X, num_clusters):
    k = num_clusters.value
    kmeans = KMeans(n_clusters=k, init='random',n_init=1, max_iter=1, random_state=0)
    y_pred = kmeans.fit_predict(X)
    return k, kmeans, y_pred


@app.cell
def __(alt, kmeans, pd):
    kmeans_centers = kmeans.cluster_centers_
    kmeans_centers = pd.DataFrame(kmeans_centers, columns=["x1", "x2"])
    points = alt.Chart(kmeans_centers).mark_point(
        filled=True,
        size=100,
        color="black",
    ).encode(
        x="x1:Q",
        y="x2:Q",
    )
    return kmeans_centers, points


@app.cell
def __(X_df, alt, mo, points, size_var, y_pred):
    X_df["cluster"] = y_pred
    chartcolored = alt.Chart(X_df).mark_point(
        filled=True,
        size=size_var,
    ).encode(
        x="x1:Q",
        y="x2:Q",
        color=alt.Color("cluster:N").title("Cluster Color")
    ).add_params(
        size_var,
    )
    chartcolored = mo.ui.altair_chart(chartcolored + points)
    return chartcolored,


@app.cell
def __(chartcolored, mo):
    mo.vstack([chartcolored, chartcolored.value.head()])
    return


@app.cell
def __(mo):
    mo.center(mo.md("##Cluster Updates"))
    return


@app.cell
def __(mo):
    # Centroid Progression
    # Maybe add the centroid update equation here?
    num_iterations = mo.ui.slider(1, 30, step=1, value=1, label='Number of Centroid Updates: ')
    num_iterations
    return num_iterations,


@app.cell
def __(mo, num_clusters):
    cluster_update_k = mo.ui.slider(1,30,step=1, value=num_clusters.value,label="Number of Clusters (k): ")
    cluster_update_k
    return cluster_update_k,


@app.cell
def __(KMeans, X, X_df, alt, cluster_update_k, mo, num_iterations, pd):
    kmeans_iterations = []
    kmeans_centroid_df = pd.DataFrame()
    for i in range(num_iterations.value):
        kmeans_it = KMeans(n_clusters=cluster_update_k.value, n_init=1, max_iter=i+1, init="random",random_state=0).fit(X)
        kmeans_iterations.append(kmeans_it)
        X_df[f"colors{i}"] = kmeans_it.predict(X)
    clusterdataframes = pd.DataFrame(columns=["x1","x2", "label"])

    index = 0
    for i in range(len(kmeans_iterations)):
        for j in range(len(kmeans_iterations[i].cluster_centers_)):
            clusterdataframes.loc[index,"x1"] = kmeans_iterations[i].cluster_centers_[j][0]
            clusterdataframes.loc[index, "x2"] = kmeans_iterations[i].cluster_centers_[j][1]
            clusterdataframes.loc[index, "label"] = kmeans_iterations[i].predict(kmeans_iterations[i].cluster_centers_)[j]
            index+=1

    size_slider_data = alt.binding_range(min=1, max=35, step=0.25, name="Data Points Size: ")
    size_var_data = alt.param(bind=size_slider_data, value=18)


    cluster_points = alt.Chart(X_df).mark_point(
        filled=True,
        size=size_var_data
    ).encode(
        x="x1:Q",
        y="x2:Q",
        color=alt.Color(f"colors{num_iterations.value-1}:N").title("Cluster Color"),
    ).add_params(
        size_var_data
    )


    lines = alt.Chart(clusterdataframes).mark_line(
        color="black",
    ).encode(
        x="x1:Q", 
        y="x2:Q",
        detail="label:N",
        strokeWidth=alt.value(4)
    )

    size_slider_centroids = alt.binding_range(min=50, max=200, step=1, name="Centroid Size: ")
    size_var_centroids = alt.param(bind=size_slider_centroids, value=80)

    center_points = alt.Chart(clusterdataframes).mark_point(
        filled=True,
        shape="cross",
        size=size_var_centroids,
        color="#FF10F0"
    ).encode(
        x="x1:Q", 
        y="x2:Q",
    ).add_params(
        size_var_centroids
    ).properties(
        width=500,
        height=400
    )

    centroid_paths = cluster_points + lines + center_points
    centroid_path = mo.ui.altair_chart(centroid_paths)
    return (
        center_points,
        centroid_path,
        centroid_paths,
        cluster_points,
        clusterdataframes,
        i,
        index,
        j,
        kmeans_centroid_df,
        kmeans_it,
        kmeans_iterations,
        lines,
        size_slider_centroids,
        size_slider_data,
        size_var_centroids,
        size_var_data,
    )


@app.cell
def __(centroid_path, mo):
    mo.vstack([centroid_path, centroid_path.value.head()])
    return


@app.cell
def __(mo):
    mo.center(mo.md("##Model Performace Measures"))
    return


@app.cell
def __(mo):
    mo.center(mo.md("###Model Inertia"))
    return


@app.cell
def __(KMeans, X, np, num_clusters, pd):
    # Plot iterias to determine what would be best k
    kmeans_per_k = [KMeans(n_clusters=k, n_init=1, max_iter=500, random_state=0).fit(X) for k in range(1,num_clusters.value+20)]
    intertias = [model.inertia_ for model in kmeans_per_k]
    inertias_df = pd.DataFrame(intertias, columns=["Inertia"])
    inertias_df["k"] = np.arange(1, num_clusters.value+20)
    inertias_df = inertias_df[["k", "Inertia"]]
    return inertias_df, intertias, kmeans_per_k


@app.cell
def __(alt, inertias_df, mo):
    inertia_chart = alt.Chart(inertias_df, title="Model Inertia as Number of Clusters Grow").mark_line(point=True).encode(
        x="k",
        y="Inertia"
    )
    inertia_charts = mo.ui.altair_chart(inertia_chart)
    return inertia_chart, inertia_charts


@app.cell
def __(inertia_charts, mo):
    mo.vstack([inertia_charts, inertia_charts.value.head()])
    return


@app.cell
def __(X, kmeans_per_k, np, num_clusters, pd, silhouette_score):
    silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]
    silhouette_df = pd.DataFrame(silhouette_scores, columns=["Silhouette score"])
    silhouette_df["k"] = np.arange(2, num_clusters.value+20)
    silhouette_df = silhouette_df[["k", "Silhouette score"]]
    return silhouette_df, silhouette_scores


@app.cell
def __(alt, mo, silhouette_df):
    silhouette_chart = alt.Chart(silhouette_df, title="Silhoutte Score with k Clusters").mark_line(point=True).encode(
        x="k",
        y="Silhouette score"
    )
    silhouette_charts = mo.ui.altair_chart(silhouette_chart)
    return silhouette_chart, silhouette_charts


@app.cell
def __(mo):
    mo.center(mo.md("###Silhoutte Score"))
    return


@app.cell
def __(mo, silhouette_charts):
    mo.vstack([silhouette_charts, silhouette_charts.value.head()])
    return


@app.cell
def __(mo):
    mo.center(mo.md("##Big O Runtimes"))
    return


@app.cell
def __(mo):
    mo.image("https://raw.githubusercontent.com/czhurdlespeed/K-MeansClustering/main/images/bigO(k).png")
    return


@app.cell
def __(mo):
    mo.image("https://raw.githubusercontent.com/czhurdlespeed/K-MeansClustering/main/images/bigO(n).png")
    return


@app.cell
def __(mo):
    mo.md("""
    ##*Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow* by Aurelien Geron
    ### Ch. 9: Unsupervised Learning Techniques
    #### Example: Image Segmentation
    """)
    return


@app.cell
def __(requests):
    ladybug_data=requests.get(   "https://raw.githubusercontent.com/czhurdlespeed/K-MeansClustering/main/images/ladybug.png")
    return ladybug_data,


@app.cell
def __(BytesIO, PIL, ladybug_data, np):
    image = np.asarray(PIL.Image.open(BytesIO(ladybug_data.content)))
    X_lady = image.reshape(-1,3)
    return X_lady, image


@app.cell
def __(mo):
    mo.image("https://raw.githubusercontent.com/czhurdlespeed/K-MeansClustering/main/images/image_segmentation_plot.png"
    )
    return


@app.cell
def __(mo):
    color_form = mo.ui.slider(1, 100, step=1, value=10, label="Lady Bug Color Range: ").form()
    color_form
    return color_form,


@app.cell
def __(KMeans, X_lady, color_form, image, mo, plt):
    segmented_imgs = []
    kmeans_lady_it = KMeans(n_clusters=color_form.value, n_init='auto', init='k-means++', random_state=42).fit(X_lady)
    segmented_img_it = kmeans_lady_it.cluster_centers_[kmeans_lady_it.labels_]
    segmented_imgs.append(segmented_img_it.reshape(image.shape))

    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_imgs[0] / 255)
    plt.title(f"Your choice: {color_form.value} color clusters")
    plt.axis('off')
    plt.savefig("yourchoice.png")
    mo.image(
        src="yourchoice.png",
        alt="lady bug color segmentation",
        width=1000,
        height=800,
        rounded=True,
    )
    return kmeans_lady_it, segmented_img_it, segmented_imgs


@app.cell
def __(mo):
    mo.center(mo.md("""
    ##**Citations**
    """))
    return


@app.cell
def __(mo):
    mo.md("""
    1. J. Watt, "K-Means Clustering*," in Machine Learning Refined. Available: https://jermwatt.github.io/machine_learning_refined/notes/8_Linear_unsupervised_learning/8_5_Kmeans.html. Accessed on: Mar. 28, 2024.
    2. A. Géron, "09_unsupervised_learning.ipynb," GitHub, Repository: "handson-ml3," Available: https://github.com/ageron/handson-ml3/blob/main/09_unsupervised_learning.ipynb. Accessed on: Mar. 28, 2024.
    3. "Clustering Algorithms: K-means," Department of Computer Science, Princeton University, Course: COS 435, Spring 2008. [Online]. Available: https://www.cs.princeton.edu/courses/archive/spr08/cos435/Class_notes/clustering2_toPost.pdf. Accessed on: Mar. 28, 2024.
    4. A. Géron, "Chapter 9: Unsupervised Learning Techniques," in Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd ed., Sebastopol, CA: O'Reilly Media, 2019.
    """)
    return


if __name__ == "__main__":
    app.run()