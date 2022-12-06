import metadata_parser
import dns.resolver
url="http://kaggle.com/"
page=metadata_parser.MetadataParser(url)
print(page.metadata)
# result=dns.resolver.query("www.kaggle.com","NS")
# for server in result:
#     print(server.target)
result = dns.resolver.query('www.youtube.com', 'A')
for ipval in result:
    print('IP:', ipval.to_text())

