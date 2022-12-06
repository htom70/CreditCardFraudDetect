FROM adoptopenjdk:14-jre-hotspot
WORKDIR /opt/app
COPY target/*.jar core.jar
CMD ["java","-jar","core.jar"]