# **Master Thesis Project**

Master Thesis project as part of a Double Master Degree at the University of Camerino and the University of Applied
Sciences of Northwestern Switzerland.

## **Abstract**

The success of research project proposals heavily depends on the **formation of an effective consortium**, ensuring the
inclusion of experienced researchers aligned with the **topics of research calls**, such as those in the EU's **Horizon
Europe** program.

One of the most challenging aspects of consortium formation is the **identification of suitable research collaborators**. Traditional approaches, which primarily rely on **social networks** and **author citations**, often fail to provide
effective recommendations.

This thesis introduces an **Agentic Graph Retrieval-Augmented Generation (RAG) method** that enhances **contextual and
explainable collaborator recommendations**. By **leveraging Knowledge Graphs (KGs) and Large Language Models (LLMs)**,
this method tailors recommendations based on researchers' **areas of expertise** and **project relevance**, surpassing
traditional approaches in effectiveness.

The **Design Science research methodology** was followed to develop this approach, and the **GPT-4o model** was used for
evaluation, demonstrating the method's capabilities in **enhancing consortium formation**.

---

## **Installation**

### Prerequisites

To run this project, ensure you have the following installed on your system:

- **Docker** (to containerize and manage services)
- **GraphDB** (for storing and querying the knowledge graph)

### Clone the Repository

```bash
  git clone https://github.com/Piermuz7/MasterThesisProject.git
```

### Streamlit Configuration

Create a `.streamlit` directory in the project root and add the following file:

```bash
  mkdir .streamlit
  touch .streamlit/secrets.toml
```

Add the following configuration to the `secrets.toml` file:

```toml
GRAPHDB_URL = "http://graphdb:7200/repositories/CORDIS-eurio-KG"
EURIO_ONTOLOGY_PATH = "ontology/EURIO.ttl"
GRAPHDB_USERNAME = "admin"
GRAPHDB_PASSWORD = "root"
AZURE_OPENAI_ENDPOINT = "<YOUR_AZURE_OPENAI_ENDPOINT>"
[api_key]
ANTHROPIC_KEY = "<YOUR_ANTHROPIC_API_KEY>"
AZURE_OPENAI_API_KEY = "<YOUR_AZURE_OPENAI_API_KEY>"
```

---

### **Importing the EURIO Knowledge Graph on GraphDB locally**

1. **Download**
   the [EURIO Knowledge Graph](https://data.europa.eu/data/datasets/named-graphs-from-eurio-knowledge-graph?locale=en)
2. **Import** it into GraphDB:
    - Open **GraphDB Workbench** (`http://localhost:7200`)
    - Create a new repository
    - Since the EURIO RDF file is large, you may need to increase the **maxInMemorySize** and **maxUploadSize**
      properties in GraphDB.
    - Go to **META-INF** GraphDB directory. On MacOS, it is located at
      `'/Applications/GraphDB Desktop.app/Contents/app/lib/common/WEB-INF/classes/META-INF'`.
   
      ```bash
      cd /Applications/GraphDB\ Desktop.app/Contents/app/lib/common/WEB-INF/classes/META-INF
      ```
    - Open properties.xml and set the following values:
        - `<entry key="graphdb.workbench.maxInMemorySize">5368709120</entry>`
        - `<entry key="graphdb.workbench.maxUploadSize">5368709120</entry>`
    - Load the **EURIO RDF N-Quad file** into the repository

- Due to the large size of the EURIO knowledge graph, it is recommended to import it locally first and then use Docker’s
  volume mapping to persist the data. This approach ensures better performance and avoids the need to repeatedly
  re-import the knowledge graph when restarting the container.

---

## Quick Start

Under the project directory, you can use the following commands to build, run, and manage the project.

### Build & Run the Project

```bash
  sudo docker-compose up --build
```

This command **builds the Docker images** and **starts all services**.
In particular, it starts the **Streamlit app**, **GraphDB**, **Chroma** vector database, and **Ollama** service.

### Restart Services

```bash
  sudo docker-compose restart
```

This restarts the **Streamlit app**, **GraphDB**, **Chroma** vector database, and **Ollama** service

### Stop Services

```bash
  sudo docker-compose stop
```

Stops all running containers without removing them.

### Stop & Remove Containers

```bash
  sudo docker-compose down
```

Stops and **removes all containers** and **networks** created by `docker-compose`.

---

## Technology Stack

| Component            | Description                                             |
|----------------------|---------------------------------------------------------|
| **Docker**           | Manages containerized services                          |
| **GraphDB**          | Stores and queries the knowledge graph                  |
| **SPARQL**           | Query language for RDF-based knowledge graphs           |
| **Python**           | Core programming language                               |
| **Streamlit**        | Provides an interactive user interface                  |
| **GPT-4o**           | Large Language Model used for RAG-based recommendations |
| **LangChain**        | Framework for integrating LLMs with external tools      |
| **LlamaIndex**       | Framework to handle agentic graph retrieval             |
| **Ollama**           | Service for generating embeddings from text             |
| **Chroma**           | Vector database for storing and querying embeddings     |
| **all-minilm:l6-v2** | sentence-transformers model for generating embeddings   |

---

## Evaluation

The evaluation of the Agentic Graph Retrieval-Augmented Generation (RAG) method was mainly conducted using the **GPT-4o** model.
Some experiments were also conducted using the **Claude 3.5 Sonnet** model.
The framework used for evaluation is **RAGAS**.
Our evaluation aims to measure the relevance and consistency of the suggested contributors, ensuring that the retrieved knowledge is in line with user demands.

The results of the artifact evaluation are presented under the `/evaluation` directory.

---

## License

This project is licensed under the **MIT License**. Feel free to modify and distribute it.

---

## Contact

For any questions or contributions, feel free to contact:  
**Piermichele Rosati**  
Email: `piermichele.rosati@gmail.com`  
GitHub: [Piermuz7](https://github.com/Piermuz7)

