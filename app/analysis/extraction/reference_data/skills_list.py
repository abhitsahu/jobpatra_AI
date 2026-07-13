"""Reference skills dataset.

This file contains ONLY data — no extraction logic whatsoever.
Each category maps a canonical skill name (as it should appear in output)
to a set of known aliases/abbreviations (all lowercased for case-insensitive
matching in skill_extractor.py).

To extend: add an entry to the relevant category dict or create a new
category key.  The extraction layer reads this at import time and needs
no changes when the data changes.

Structure:
    SKILLS: dict[str, dict[str, frozenset[str]]]
    - top-level key   → category label shown in output
    - second-level key → canonical skill name (display form)
    - value           → frozenset of lowercase aliases including the canonical
"""

# ---------------------------------------------------------------------------
# Canonical → aliases mapping, grouped by category.
# Key   = canonical display name  (used in output)
# Value = frozenset of lowercase aliases the extractor matches against
# ---------------------------------------------------------------------------

SKILLS: dict[str, dict[str, frozenset[str]]] = {
    "Programming Languages": {
        "Python": frozenset({"python", "py"}),
        "JavaScript": frozenset({"javascript", "js", "ecmascript"}),
        "TypeScript": frozenset({"typescript", "ts"}),
        "Java": frozenset({"java"}),
        "Go": frozenset({"go", "golang"}),
        "Rust": frozenset({"rust"}),
        "C++": frozenset({"c++", "cpp", "c plus plus"}),
        "C#": frozenset({"c#", "csharp", "c sharp"}),
        "PHP": frozenset({"php"}),
        "Ruby": frozenset({"ruby", "rb"}),
        "Swift": frozenset({"swift"}),
        "Kotlin": frozenset({"kotlin"}),
        "Scala": frozenset({"scala"}),
        "R": frozenset({"r language", "r programming"}),
        "Shell": frozenset({"shell", "bash", "sh", "zsh", "fish"}),
    },
    "Frontend": {
        "React": frozenset({"react", "reactjs", "react.js"}),
        "Vue.js": frozenset({"vue", "vuejs", "vue.js"}),
        "Angular": frozenset({"angular", "angularjs", "angular.js"}),
        "Next.js": frozenset({"next.js", "nextjs", "next"}),
        "Nuxt.js": frozenset({"nuxt", "nuxtjs", "nuxt.js"}),
        "Svelte": frozenset({"svelte", "sveltejs"}),
        "HTML": frozenset({"html", "html5"}),
        "CSS": frozenset({"css", "css3"}),
        "Tailwind CSS": frozenset({"tailwind", "tailwindcss", "tailwind css"}),
        "Bootstrap": frozenset({"bootstrap"}),
        "SASS": frozenset({"sass", "scss"}),
        "Redux": frozenset({"redux"}),
        "GraphQL": frozenset({"graphql", "gql"}),
        "WebSockets": frozenset({"websockets", "websocket", "ws"}),
    },
    "Backend": {
        "Node.js": frozenset({"node", "nodejs", "node.js"}),
        "FastAPI": frozenset({"fastapi"}),
        "Django": frozenset({"django"}),
        "Flask": frozenset({"flask"}),
        "Express.js": frozenset({"express", "expressjs", "express.js"}),
        "Spring Boot": frozenset({"spring", "spring boot", "springboot"}),
        "NestJS": frozenset({"nest", "nestjs", "nest.js"}),
        "Laravel": frozenset({"laravel"}),
        "Rails": frozenset({"rails", "ruby on rails", "ror"}),
        "ASP.NET": frozenset({"asp.net", "aspnet", "asp net"}),
        "REST API": frozenset({"rest", "rest api", "restful", "restful api"}),
        "gRPC": frozenset({"grpc"}),
    },
    "Databases": {
        "PostgreSQL": frozenset({"postgresql", "postgres", "psql"}),
        "MySQL": frozenset({"mysql"}),
        "MongoDB": frozenset({"mongodb", "mongo"}),
        "Redis": frozenset({"redis"}),
        "SQLite": frozenset({"sqlite"}),
        "Elasticsearch": frozenset({"elasticsearch", "elastic search", "es"}),
        "Cassandra": frozenset({"cassandra", "apache cassandra"}),
        "DynamoDB": frozenset({"dynamodb", "dynamo db"}),
        "Firebase": frozenset({"firebase", "firestore"}),
        "SQL": frozenset({"sql"}),
        "NoSQL": frozenset({"nosql", "no-sql"}),
    },
    "Cloud": {
        "AWS": frozenset({"aws", "amazon web services"}),
        "GCP": frozenset({"gcp", "google cloud", "google cloud platform"}),
        "Azure": frozenset({"azure", "microsoft azure"}),
        "Heroku": frozenset({"heroku"}),
        "Vercel": frozenset({"vercel"}),
        "Netlify": frozenset({"netlify"}),
        "DigitalOcean": frozenset({"digitalocean", "digital ocean"}),
        "Cloudflare": frozenset({"cloudflare"}),
        "Lambda": frozenset({"lambda", "aws lambda"}),
        "S3": frozenset({"s3", "aws s3"}),
        "EC2": frozenset({"ec2", "aws ec2"}),
    },
    "DevOps": {
        "Docker": frozenset({"docker", "dockerfile"}),
        "Kubernetes": frozenset({"kubernetes", "k8s", "k 8 s"}),
        "CI/CD": frozenset({"ci/cd", "cicd", "ci cd", "continuous integration", "continuous delivery"}),
        "GitHub Actions": frozenset({"github actions", "gh actions"}),
        "Jenkins": frozenset({"jenkins"}),
        "Terraform": frozenset({"terraform"}),
        "Ansible": frozenset({"ansible"}),
        "Nginx": frozenset({"nginx"}),
        "Linux": frozenset({"linux", "ubuntu", "debian", "centos"}),
        "Git": frozenset({"git", "github", "gitlab", "bitbucket"}),
    },
    "Data & ML": {
        "Machine Learning": frozenset({"machine learning", "ml"}),
        "Deep Learning": frozenset({"deep learning", "dl"}),
        "TensorFlow": frozenset({"tensorflow", "tf"}),
        "PyTorch": frozenset({"pytorch", "torch"}),
        "Pandas": frozenset({"pandas"}),
        "NumPy": frozenset({"numpy", "np"}),
        "Scikit-learn": frozenset({"scikit-learn", "sklearn", "scikit learn"}),
        "Jupyter": frozenset({"jupyter", "jupyter notebook", "jupyter lab"}),
        "Spark": frozenset({"spark", "apache spark", "pyspark"}),
        "Kafka": frozenset({"kafka", "apache kafka"}),
        "Airflow": frozenset({"airflow", "apache airflow"}),
        "dbt": frozenset({"dbt", "data build tool"}),
    },
    "Tools": {
        "VS Code": frozenset({"vscode", "vs code", "visual studio code"}),
        "Postman": frozenset({"postman"}),
        "Jira": frozenset({"jira"}),
        "Confluence": frozenset({"confluence"}),
        "Figma": frozenset({"figma"}),
        "Swagger": frozenset({"swagger", "openapi"}),
        "Jest": frozenset({"jest"}),
        "Pytest": frozenset({"pytest"}),
        "Webpack": frozenset({"webpack"}),
        "Vite": frozenset({"vite"}),
        "npm": frozenset({"npm"}),
        "yarn": frozenset({"yarn"}),
    },
}
