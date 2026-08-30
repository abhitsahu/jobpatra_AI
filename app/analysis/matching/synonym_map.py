"""Synonym map — the single source of truth for keyword aliases.

This file contains ONLY data. No extraction logic. No matching logic.
Each entry maps a canonical term (used as the display label in output) to a
frozenset of lowercased synonyms that should be considered equivalent to it.

Rules for maintainers
---------------------
- Keys are the **canonical** form (how the term should appear in reports).
- Values are **all** known aliases, including the canonical itself (lowercased).
- All values MUST be lowercased — matching code relies on this invariant.
- Group related synonyms on adjacent lines for readability.
- Do NOT add matching logic to this file.

Adding a new synonym group
--------------------------
    "Vue.js": frozenset({"vue.js", "vue", "vuejs"}),
"""

# Synonym groups
# canonical → frozenset of lowercase aliases (including canonical itself)

SYNONYMS: dict[str, frozenset[str]] = {
    # JavaScript ecosystem 
    "JavaScript": frozenset({"javascript", "js", "ecmascript", "es6", "es2015", "es2016",
                              "es2017", "es2018", "es2019", "es2020"}),
    "TypeScript": frozenset({"typescript", "ts"}),
    "Node.js":    frozenset({"node.js", "node", "nodejs"}),
    "React":      frozenset({"react", "react.js", "reactjs"}),
    "Vue.js":     frozenset({"vue.js", "vue", "vuejs"}),
    "Angular":    frozenset({"angular", "angularjs", "angular.js"}),
    "Next.js":    frozenset({"next.js", "nextjs", "next"}),
    "Express.js": frozenset({"express.js", "express", "expressjs"}),

    #  Python ecosystem 
    "Python":      frozenset({"python", "py"}),
    "Django":      frozenset({"django"}),
    "Flask":       frozenset({"flask"}),
    "FastAPI":     frozenset({"fastapi"}),
    "Pandas":      frozenset({"pandas", "pd"}),
    "NumPy":       frozenset({"numpy", "np"}),
    "Scikit-learn":frozenset({"scikit-learn", "sklearn", "scikit learn"}),

    #  Databases 
    "PostgreSQL":     frozenset({"postgresql", "postgres", "psql"}),
    "MySQL":          frozenset({"mysql"}),
    "MongoDB":        frozenset({"mongodb", "mongo"}),
    "Redis":          frozenset({"redis"}),
    "Elasticsearch":  frozenset({"elasticsearch", "elastic search", "es", "elastic"}),
    "SQLite":         frozenset({"sqlite", "sqlite3"}),

    #  Cloud 
    "AWS":          frozenset({"aws", "amazon web services", "amazon aws"}),
    "GCP":          frozenset({"gcp", "google cloud", "google cloud platform"}),
    "Azure":        frozenset({"azure", "microsoft azure", "ms azure"}),
    "AWS EC2":      frozenset({"aws ec2", "ec2", "amazon ec2"}),
    "AWS S3":       frozenset({"aws s3", "s3", "amazon s3"}),
    "AWS Lambda":   frozenset({"aws lambda", "lambda"}),

    #  DevOps / Infrastructure 
    "Docker":       frozenset({"docker", "dockerfile"}),
    "Docker Compose":frozenset({"docker compose", "docker-compose", "docker_compose"}),
    "Kubernetes":   frozenset({"kubernetes", "k8s"}),
    "Terraform":    frozenset({"terraform", "tf"}),
    "Ansible":      frozenset({"ansible"}),
    "Jenkins":      frozenset({"jenkins"}),
    "CI/CD":        frozenset({"ci/cd", "cicd", "ci cd", "continuous integration",
                               "continuous delivery", "continuous deployment", "github actions",
                               "jenkins"}),
    "GitHub Actions":frozenset({"github actions", "gh actions", "github-actions"}),

    # Payment / Fintech Domain
    "Functional Programming": frozenset({
        "functional programming", "haskell", "scala", "python fp",
        "lambda calculus", "immutability",
    }),
    "Distributed Systems": frozenset({
        "distributed systems", "microservices", "scalability", "concurrency",
        "parallel computing", "service mesh",
    }),
    "Payment Orchestration": frozenset({
        "payment orchestration", "transaction routing", "payment flow",
        "orchestration layer", "payment gateway",
    }),
    "Multi-DC Architecture": frozenset({
        "multi-dc architecture", "multi-datacenter", "geo-redundant",
        "high availability", "disaster recovery",
    }),
    "Self-Healing Systems": frozenset({
        "self-healing systems", "auto-recovery", "fault tolerance",
        "resilience engineering",
    }),
    "Traffic Routing": frozenset({
        "traffic routing", "load balancing", "request routing", "api gateway", "ingress",
    }),
    "Anomaly Detection": frozenset({
        "anomaly detection", "fraud detection", "outlier detection",
        "real-time monitoring", "alerting",
    }),
    "Payment Tokenization": frozenset({
        "payment tokenization", "tokenization", "data masking", "vaulting", "secure storage",
    }),
    "Fraud & Risk Management": frozenset({
        "fraud & risk management", "fraud prevention", "risk scoring", "compliance", "aml",
    }),
    "Edge Computing": frozenset({"edge computing", "edge devices", "cdn", "distributed edge"}),
    "First Principles Thinking": frozenset({
        "first principles thinking", "fundamental reasoning", "principles-based design",
        "system design",
    }),
    "Low-Code/No-Code": frozenset({
        "low-code/no-code", "rapid application development", "visual programming",
        "citizen development",
    }),
    "API Integrations": frozenset({
        "api integrations", "rest apis", "graphql", "webhooks", "sdk integration",
    }),
    "Infrastructure as Code": frozenset({
        "infrastructure as code", "terraform", "cloudformation", "pulumi", "iac",
    }),

    #  Data & ML 
    "Machine Learning": frozenset({"machine learning", "ml"}),
    "Deep Learning":    frozenset({"deep learning", "dl"}),
    "TensorFlow":       frozenset({"tensorflow", "tf"}),
    "PyTorch":          frozenset({"pytorch", "torch"}),

    #  Other languages 
    "Go":      frozenset({"go", "golang"}),
    "Rust":    frozenset({"rust"}),
    "Java":    frozenset({"java"}),
    "Kotlin":  frozenset({"kotlin"}),
    "Swift":   frozenset({"swift"}),
    "C++":     frozenset({"c++", "cpp"}),
    "C#":      frozenset({"c#", "csharp"}),
    "PHP":     frozenset({"php"}),
    "Ruby":    frozenset({"ruby", "rb"}),
    "Scala":   frozenset({"scala"}),
    "Shell":   frozenset({"shell", "bash", "sh", "zsh"}),

    #  Tools 
    "Git":         frozenset({"git", "github", "gitlab", "bitbucket"}),
    "GraphQL":     frozenset({"graphql", "gql"}),
    "REST API":    frozenset({"rest", "rest api", "restful", "restful api", "http api"}),
    "gRPC":        frozenset({"grpc"}),
    "Spring Boot": frozenset({"spring boot", "spring", "springboot"}),
    "Kafka":       frozenset({"kafka", "apache kafka"}),
    "Spark":       frozenset({"spark", "apache spark", "pyspark"}),
    "Nginx":       frozenset({"nginx"}),
    "Linux":       frozenset({"linux", "ubuntu", "debian", "centos"}),
}
