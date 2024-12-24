import os
import re
from datetime import datetime
from pathlib import Path

from django.conf import settings  # Import settings to use BASE_DIR
from django.shortcuts import render

from .letor import rerank_search_results


def index(request):
    start_time = datetime.now()
    totalTime = None
    sumDocs = None
    # Get num_results from request, default to 30 if not specified
    num_results = int(request.GET.get('num_results', 30))

    query = request.GET.get('search-bar')

    if not query:
        context = {
            'query': query,
            'flag': 0,
            'num_results': num_results  # Pass to template for form persistence
        }
        return render(request, 'search/index.html', context)

    else:
        result = {}

        try:
            # Pass the num_results to rerank_search_results
            sorted_did_scores = rerank_search_results(
                query,
                top_k=num_results  # Limit results to user requested amount
            )
        except FileNotFoundError as e:
            context = {
                'query': query,
                'flag': 3,
                'error_message': str(e),
                'num_results': num_results
            }
            return render(request, 'search/index.html', context)

        # Limit the results processing to num_results
        for score, doc in sorted_did_scores[:num_results]:
            path_doc = Path(settings.BASE_DIR) / 'your_app' / doc.lstrip('.')

            if not os.path.isfile(path_doc):
                result[f"Missing Document: {doc}"] = "Content not available."
                continue

            try:
                with path_doc.open(encoding='utf-8') as file:
                    content = file.read()

                parts = path_doc.parts

                if len(parts) >= 3:
                    collection_number = parts[-2]
                    document_id_with_ext = parts[-1]
                    document_id = Path(document_id_with_ext).stem

                    collection_info = f"Collections {collection_number}: {document_id_with_ext}"
                else:
                    collection_info = f"Document: {path_doc.name}"

                result[collection_info] = content

            except Exception as e:
                result[f"Error Reading Document: {doc}"] = f"Error: {str(e)}"

        if not result:
            totalTime = 0
            sumDocs = 0

            context = {
                'query': query,
                'flag': 1,
                'totalTime': totalTime,
                'sumDocs': sumDocs,
                'num_results': num_results
            }

            return render(request, 'search/index.html', context)

        end_time = datetime.now()
        totalTime = (end_time - start_time).total_seconds()

        sumDocs = len(result)

        context = {
            'query': query,
            'flag': 2,
            'result': result,
            'totalTime': totalTime,
            'sumDocs': sumDocs,
            'num_results': num_results  # Pass to template for form persistence
        }

        return render(request, 'search/index.html', context)


# views.py
def isi(request, doc):
    # Get num_results from request, default to 30
    num_results = request.GET.get('num_results', '30')
    
    title = doc
    content = None
    digits = re.findall(r'\d+', doc)

    url_path = os.path.dirname(__file__) + '/collections/' + '/'.join(digits) + '.txt'
    with open(url_path, encoding='utf-8') as file:
        for line in file:
            content = line

    title1 = title.split(":")[1]
    title2 = title.split(":")[0]

    context = {
        'title': title,
        'title1': title1,
        'title2': title2,
        'content': content,
        'num_results': num_results,  # Add num_results to context
        'query': request.GET.get('search-bar', '')  # Preserve search query if it exists
    }

    return render(request, 'search/isi.html', context)