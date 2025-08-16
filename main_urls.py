"""
Main URL configuration for the portfolio with integrated EEG dashboard
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from django.shortcuts import render

def portfolio_home(request):
    """Serve the main portfolio page"""
    with open('index.html', 'r') as f:
        content = f.read()
    return HttpResponse(content)

def serve_asset(request, path):
    """Serve static assets like CSS, JS, images"""
    import mimetypes
    import os
    
    file_path = os.path.join('assets', path)
    if os.path.exists(file_path):
        content_type, _ = mimetypes.guess_type(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()
        return HttpResponse(content, content_type=content_type)
    else:
        return HttpResponse("Not Found", status=404)

urlpatterns = [
    # Portfolio home page
    path('', portfolio_home, name='portfolio_home'),
    
    # EEG Dashboard routes
    path('eeg-dashboard/', include('dashboard.urls')),
    
    # Django admin (optional)
    path('admin/', admin.site.urls),
    
    # Serve portfolio assets
    path('assets/<path:path>', serve_asset, name='serve_asset'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])