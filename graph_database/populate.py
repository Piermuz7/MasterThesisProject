from SPARQLWrapper import SPARQLWrapper, JSON


class GraphDatabasePopulator:
    def __init__(self, eurio_endpoint, local_graphdb_endpoint):
        """
        Initialize connections for both EURIO and local GraphDB SPARQL endpoints.
        :param eurio_endpoint: The URL of the EURIO SPARQL endpoint.
        :param local_graphdb_endpoint: The URL of your local GraphDB SPARQL endpoint.
        """
        if not eurio_endpoint or not local_graphdb_endpoint:
            raise ValueError("Both EURIO and local GraphDB SPARQL endpoints must be provided.")
        self.eurio_sparql = SPARQLWrapper(eurio_endpoint)
        self.local_graphdb_sparql = SPARQLWrapper(local_graphdb_endpoint)

    def fetch_data_from_eurio(self, project_name):
        """
        Execute a SPARQL query on the EURIO endpoint to fetch data based on a project name.
        :param project_name: The project name (eurio:title) to filter the SPARQL query.
        :return: Query results as a list of dictionaries.
        """
        # SPARQL query with project name as a parameter
        query = f"""
        PREFIX eurio: <http://data.europa.eu/s66#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?participant_name ?country_name
        WHERE {{
          ?project a eurio:Project .
          ?project eurio:title "{project_name}" .
          ?project eurio:hasInvolvedParty ?role .
          ?role eurio:isRoleOf ?o.
          ?o eurio:hasSite ?site.
          ?o eurio:legalName ?participant_name.
          ?role eurio:roleLabel "participant" .
          ?site eurio:hasGeographicalLocation ?country.
          ?country eurio:name ?country_name.
          ?country eurio:hasISOCountryCode ?iso.
        }}
        """
        self.eurio_sparql.setQuery(query)
        self.eurio_sparql.setReturnFormat(JSON)
        results = self.eurio_sparql.query().convert()
        print(results)

        # Parse the results into a list of dictionaries
        '''data = []
        for result in results["results"]["bindings"]:
            data.append({key: value["value"] for key, value in result.items()})
        return data'''
        return []

    def insert_data_into_local_graphdb(self, insert_query):
        """
        Insert data into the local GraphDB SPARQL endpoint using a query.
        :param insert_query: The SPARQL INSERT query.
        """
        self.local_graphdb_sparql.setQuery(insert_query)
        self.local_graphdb_sparql.method = 'POST'
        self.local_graphdb_sparql.query()

    def populate_from_project_name(self, project_name, transform_fn, insert_template):
        """
        Fetch data from EURIO and populate the local GraphDB database.
        :param project_name: The project name to filter data.
        :param transform_fn: A function to transform the fetched data into an insertion format.
        :param insert_template: A function or template to create insertion queries for the graph database.
        """
        # Fetch data using the project name from EURIO
        data = self.fetch_data_from_eurio(project_name)

        # Transform and insert data into the local GraphDB
        for item in data:
            print(item)
            # transformed_data = transform_fn(item)
            # insert_query = insert_template(transformed_data)
            # self.insert_data_into_local_graphdb(insert_query)


# Example transformation function
def transform_sparql_to_graph(data_item):
    """
    Transform a SPARQL query result item into the format needed for graph database insertion.
    :param data_item: A dictionary representing a single query result item.
    :return: Transformed data.
    """
    return {
        "organisation_name": data_item.get("participant_name"),
        "country_name": data_item.get("country_name"),
    }


# Example SPARQL INSERT template function
def sparql_insert_template(data):
    """
    Generate a SPARQL INSERT query based on transformed data.
    :param data: Transformed data as a dictionary.
    :return: SPARQL INSERT query string.
    """
    return f"""
    PREFIX eurio: <http://data.europa.eu/s66#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    INSERT DATA {{
      eurio:{data['organisation_name'].replace(' ', '_')} a eurio:Organisation ;
                               eurio:hasName "{data['organisation_name']}"^^xsd:string ;
                               eurio:hasCountry "{data['country_name']}"^^xsd:string .
    }}
    """


# Example Usage
if __name__ == "__main__":
    # Initialize the database populator
    populator = GraphDatabasePopulator(
        eurio_endpoint="http://data.europa.eu/s66",
        local_graphdb_endpoint="http://localhost:7200/repositories/thesis"
    )

    # Project name as input
    project_name = "Knowledge-Based Information Agent with Social Competence and Human Interaction Capabilities"

    # Fetch and populate using the project name
    populator.populate_from_project_name(
        project_name=project_name,
        transform_fn=transform_sparql_to_graph,
        insert_template=sparql_insert_template
    )
