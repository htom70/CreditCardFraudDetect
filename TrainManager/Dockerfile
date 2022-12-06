FROM adoptopenjdk:14-jre-hotspot
WORKDIR /opt/app
COPY target/*.jar trainmanager.jar
CMD ["java","-jar","trainmanager.jar"]