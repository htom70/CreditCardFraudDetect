FROM adoptopenjdk:14-jre-hotspot
WORKDIR /opt/app
COPY target/*.jar coremanager.jar
CMD ["java","-jar","coremanager.jar"]