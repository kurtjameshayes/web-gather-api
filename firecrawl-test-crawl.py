from firecrawl import Firecrawl

firecrawl = Firecrawl(api_key="fc-833609f9fa5c4540937bd7b43ca4dd35")

docs = firecrawl.crawl(url="https://www.caprivacy.org/cpra-text/", limit=2)
print(docs)

