from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pipeline import prepare_pipeline,fetch_dataset

true = True
false = False

def main():
    dataset = fetch_dataset(dataset_id=320, quiet=false)  # Fetching the dataset with ID 320
    transformed_data = prepare_pipeline(dataset)

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(transformed_data)
    labels = kmeans.labels_
    pca = PCA(n_components=2).fit_transform(transformed_data)

    plt.figure(figsize=(12, 8))
    plt.scatter(pca[:, 0], pca[:, 1], c=labels, cmap='Set1', s=50)
    plt.xlabel('Academic and Social Composite (PCA Component 1)') # ترکیب آکادمیک و اجتماعی
    plt.ylabel('Behavioral and Demographic Composite (PCA Component 2)') # ترکیب رفتاری و جمعیتی
    plt.grid(true)
    plt.show()


if __name__ == "__main__":
    main()
