import os
import re
from datetime import datetime
from pathlib import Path


from django.shortcuts import render

from .letor import rerank_search_results
from django.conf import settings  # Import settings to use BASE_DIR


def index(request):
    start_time = datetime.now()
    totalTime = None
    sumDocs = None
    num_results = request.GET.get('num_results', 30)  # Default is 30

    query = request.GET.get('search-bar')

    if not query:
        context = {
            'query': query,
            'flag': 0
        }
        return render(request, 'search/index.html', context)

    else:
        result = {}

        try:
            # Ensure rerank_search_results uses the correct model directory
            sorted_did_scores = rerank_search_results(
                query  # Adjust 'your_app' accordingly
            )
        except FileNotFoundError as e:
            # Handle the case where models are not trained yet
            context = {
                'query': query,
                'flag': 3,  # New flag indicating models are not available
                'error_message': str(e)
            }
            return render(request, 'search/index.html', context)

        for score, doc in sorted_did_scores:
            # Construct the absolute path correctly using os.path.join
            # Assume 'doc' is something like 'collections/7/79903.txt'
            # path_doc = os.path.join(settings.BASE_DIR, 'your_app', doc.lstrip('.'))
            path_doc = Path(settings.BASE_DIR) / 'your_app' / doc.lstrip('.')
            # Debugging: Print or log the constructed path
            # print(f"Opening document: {path_doc}")

            # Check if the file exists before attempting to open
            if not os.path.isfile(path_doc):
                # Handle missing files gracefully
                result[f"Missing Document: {doc}"] = "Content not available."
                continue

            try:
                # Read the entire content or handle as needed
                with path_doc.open(encoding='utf-8') as file:
                    content = file.read()

                # Use pathlib to split the path and extract collection info
                parts = path_doc.parts  # This splits the path into its components

                # Ensure the path has at least three parts: collections, collection number, and document ID
                if len(parts) >= 3:
                    collection_number = parts[-2]  # Second last part: '0'
                    document_id_with_ext = parts[-1]  # Last part: '54422.txt'
                    document_id = Path(document_id_with_ext).stem  # '54422'

                    collection_info = f"Collections {collection_number}: {document_id_with_ext}"
                else:
                    # Handle unexpected path formats
                    collection_info = f"Document: {path_doc.name}"

                result[collection_info] = content

            except Exception as e:
                # Handle any other exceptions during file reading
                result[f"Error Reading Document: {doc}"] = f"Error: {str(e)}"

        if not result:
            totalTime = 0
            sumDocs = 0

            context = {
                'query': query,
                'flag': 1,
                'totalTime': totalTime,
                'sumDocs': sumDocs
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
            'sumDocs': sumDocs
        }

        return render(request, 'search/index.html', context)


def isi(request, doc):
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
    }

    return render(request, 'search/isi.html', context)
