FROM adoptopenjdk:14-jre-hotspot
WORKDIR /opt/app
COPY target/*.jar simulator.jar
CMD ["java","-jar","simulator.jar"]