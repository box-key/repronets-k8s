FROM node:16

WORKDIR /usr/app

# copy source files
COPY package.json /usr/app/
COPY pages/ /usr/app/pages/
COPY public/ /usr/app/public/
COPY styles/ /usr/app/styles/

# install
RUN npm install

# build 
RUN npm run build

ENTRYPOINT [ "npm", "run", "start" ]
 
