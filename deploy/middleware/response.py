class Response:

    def __init__(self, get_response):

        self.get_response = get_response

    def __call__(self, request):

        response = self.get_response(request)

        print(response.content)

        return response

# x = Response(request)
# print(x(request))

        