"""itsp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import url
from django.urls import path
from latex.views import (
     predict, 
     index, 
     viewDatabase,
     pdf, 
     # viewAbout, 
     # viewPdf,
    )
# from django.http import 
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
# from latex import views as web_views

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$', index, name='homepage'),
    url('predict', predict, name='predict'),
    url('gallery', viewDatabase, name='gallery'),
    url('pdf', pdf, name='pdf'),
    # path(
    #     "pdf",
    #     web_views.print_pdf,
    #     name="print_pdf",
    # ),

    # url('team', viewAbout, name='team'),
    # url('pdf', viewPdf, name='pdf'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += staticfiles_urlpatterns()



