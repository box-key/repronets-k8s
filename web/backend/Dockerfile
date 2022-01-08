FROM node:16

WORKDIR /usr/src/app

COPY package.json /usr/src/app/
COPY src/ /usr/src/app/src/

RUN npm install

ENTRYPOINT [ "npm", "start" ]
